"""Treina U-Net, SegFormer e Mask2Former para cada pré-processamento (PROTOCOLO.MD, Seção 7).

- 20 épocas fixas; ao fim de cada época, Dice na validação (subconjunto fixo) e checkpoint da
  melhor época (``{arch}_{pre}_best_model.pt``), com a época registrada.
- As probabilidades de validação, teste e do conjunto visual são salvas com o melhor checkpoint.

Uso:
    python scripts/05_treinar.py --arch all --pre all            # 33 treinos (3 x 11)
    python scripts/05_treinar.py --arch unet segformer --pre all # subconjunto (ex.: uma GPU)
    python scripts/05_treinar.py --bench                         # mede s/iteração e estima o tempo total
    CUDA_VISIBLE_DEVICES=0 python scripts/05_treinar.py --fila & CUDA_VISIBLE_DEVICES=1 python scripts/05_treinar.py --fila
    python scripts/05_treinar.py --arch all --pre P0 P1 P6 P9 --piloto   # piloto em CPU

Saídas (``results/exp`` ou ``results/piloto``):
    runs/{arch}_{pre}.json             configuração, histórico por época, melhor época, custo
    probs/{arch}_{pre}_{split}.npz     probabilidades float16 (n_traj, n_t, H, W)
    checkpoints/{arch}_{pre}_best_model.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import models, preprocessing  # noqa: E402
from src.data.experiment import ROOT, FrameDataset, load_config, load_split, predict_frames  # noqa: E402
from src.evaluation.metrics import confusion, scores  # noqa: E402
from src.ground_truth.generation import TAU, gt_mask, valid_frames  # noqa: E402


def subset_piloto(d: dict, n_traj: int, every: int) -> dict:
    """Piloto: as primeiras n_traj trajetórias por regime e 1 de cada ``every`` frames."""
    keep = np.concatenate([np.nonzero(d["regime"] == r)[0][:n_traj] for r in dict.fromkeys(d["regime"])])
    out = {k: d[k][keep] for k in ("A", "B", "regime", "traj")}
    out["A"], out["B"] = out["A"][:, ::every], out["B"][:, ::every]
    out["times"] = d["times"][::every]
    return out


def make_opt(model, cfg, arch, steps):
    lr = cfg["treino"]["lr"][arch]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["treino"]["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    return opt, sched


def train_step(model, x, y, opt, sched, scaler, dev, amp):
    with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
        loss = model.loss(x, y)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    sched.step()
    return loss.item()


def bench(cfg, archs, dev, amp, n_train, n_val, n_eval):
    """Mede s/iteração no dispositivo e estima o tempo de um treino completo por arquitetura."""
    for arch in archs:
        torch.manual_seed(0)
        model = models.build(arch, pretrained=cfg["treino"]["pretreinado"],
                             mask2former_points=cfg["treino"]["mask2former_pontos"]).to(dev)
        bs = cfg["treino"]["batch"][arch]
        opt, sched = make_opt(model, cfg, arch, 1000)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        x = torch.rand(bs, 1, 128, 128, device=dev)
        y = (torch.rand(bs, 128, 128, device=dev) > 0.7).float()
        model.train()
        for i in range(13):
            if i == 3:
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
            loss = train_step(model, x, y, opt, sched, scaler, dev, amp)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        s_it = (time.perf_counter() - t0) / 10
        model.eval()
        t0 = time.perf_counter()
        predict_frames(model, np.random.rand(64, 128, 128).astype(np.float32), dev, batch=32, amp=amp)
        s_frame = (time.perf_counter() - t0) / 64
        epoch = s_it * n_train / bs + s_frame * n_val
        total = cfg["treino"]["epocas"] * epoch + s_frame * n_eval
        print(f"[bench] {arch}: {s_it:.3f} s/it (lote {bs}), {1000 * s_frame:.1f} ms/frame inferência, "
              f"perda finita={np.isfinite(loss)}; ~{epoch / 60:.1f} min/época, ~{total / 60:.0f} min por treino, "
              f"~{11 * total / 3600:.1f} h para 11 pré", flush=True)
        del model, opt
        if dev.type == "cuda":
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", nargs="+", default=["all"])
    ap.add_argument("--pre", nargs="+", default=["all"])
    ap.add_argument("--epocas", type=int, default=None)
    ap.add_argument("--piloto", action="store_true", help="dados e épocas reduzidos (validação do pipeline em CPU)")
    ap.add_argument("--amp", action="store_true", help="precisão mista fp16 (só CUDA)")
    ap.add_argument("--bench", action="store_true", help="só mede a velocidade e estima o tempo total")
    ap.add_argument("--refazer", action="store_true")
    ap.add_argument("--fila", action="store_true",
                    help="fila compartilhada entre processos (várias GPUs): cada treino é reservado por um arquivo .lock")
    args = ap.parse_args()

    cfg = load_config()
    archs = list(models.ARCHS) if args.arch == ["all"] else args.arch
    pres = preprocessing.PRE_IDS if args.pre == ["all"] else args.pre
    out_dir = ROOT / ("results/piloto" if args.piloto else "results/exp")
    epochs = args.epocas or (cfg["treino"]["epocas"] if not args.piloto else 4)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.amp and dev.type == "cuda"
    pretrained = cfg["treino"]["pretreinado"]

    data = {s: load_split(cfg, s) for s in ("train", "valid", "test", "visual")}
    if args.piloto:
        data["train"] = subset_piloto(data["train"], 4, 1)
        data["valid"] = subset_piloto(data["valid"], 2, 5)
        data["test"] = subset_piloto(data["test"], 3, 5)
    for d in data.values():
        d["M"] = gt_mask(d["B"])
        d["ok"] = valid_frames(d["B"], d["times"])

    tr, va = data["train"], data["valid"]
    A_fit, M_fit = tr["A"][tr["ok"]], tr["M"][tr["ok"]]
    A_va, M_va = va["A"][va["ok"]], va["M"][va["ok"]]
    sel = np.linspace(0, len(A_va) - 1, min(cfg["treino"]["validacao_por_epoca"], len(A_va))).astype(int)
    n_eval = sum(d["A"].shape[0] * d["A"].shape[1] for k, d in data.items() if k != "train")

    if args.bench:
        print(f"[bench] dispositivo {dev} ({torch.cuda.get_device_name() if dev.type == 'cuda' else platform.processor()}), "
              f"amp={amp}, {len(A_fit)} frames de treino", flush=True)
        bench(cfg, archs, dev, amp, len(A_fit), len(sel), n_eval)
        return

    # Mais caro primeiro: com --fila, as GPUs livres pegam o próximo treino disponível.
    jobs = [(a, p) for a in sorted(archs, key=lambda a: a != "mask2former") for p in pres]
    claims = out_dir / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    for arch, pid in jobs:
            tag = f"{arch}_{pid}"
            run_path = out_dir / "runs" / f"{tag}.json"
            if run_path.exists() and not args.refazer:
                print(f"[{tag}] já existe, pulando", flush=True)
                continue
            if args.fila:
                try:
                    os.close(os.open(claims / f"{tag}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                except FileExistsError:
                    continue  # outro processo já está treinando este
            torch.manual_seed(cfg["seed"])
            np.random.seed(cfg["seed"])
            pre = preprocessing.build(pid).fit(A_fit)
            t0 = time.time()
            Xtr = pre(A_fit)
            pre_ms_frame = 1000 * (time.time() - t0) / len(Xtr)
            Xsel = pre(A_va[sel])

            model = models.build(arch, pretrained=pretrained,
                                 mask2former_points=cfg["treino"]["mask2former_pontos"]).to(dev)
            n_params = sum(p.numel() for p in model.parameters())
            bs = cfg["treino"]["batch"][arch]
            dl = torch.utils.data.DataLoader(
                FrameDataset(Xtr, M_fit, train=True), batch_size=bs, shuffle=True,
                num_workers=cfg["treino"]["workers"], drop_last=True, pin_memory=dev.type == "cuda",
                generator=torch.Generator().manual_seed(cfg["seed"]))
            opt, sched = make_opt(model, cfg, arch, epochs * len(dl))
            scaler = torch.amp.GradScaler("cuda", enabled=amp)
            hist, best, best_state = [], (-1.0, 0), None
            t_train = time.time()
            for ep in range(1, epochs + 1):
                model.train()
                tot, n = 0.0, 0
                for x, y in dl:
                    x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                    tot += train_step(model, x, y, opt, sched, scaler, dev, amp) * len(x)
                    n += len(x)
                model.eval()
                P = predict_frames(model, Xsel, dev, batch=32, amp=amp)
                s = scores(confusion(P > 0.5, M_va[sel]).sum(0))
                hist.append({"epoca": ep, "perda_treino": tot / n, "dice_valid": float(s["Dice"]),
                             "iou_valid": float(s["IoU"]), "tempo_s": time.time() - t_train})
                if s["Dice"] > best[0]:
                    best = (float(s["Dice"]), ep)
                    best_state = copy.deepcopy(model.state_dict())
                print(f"[{tag}] época {ep}/{epochs} perda {tot / n:.4f} Dice valid {s['Dice']:.4f} "
                      f"(melhor {best[0]:.4f} na época {best[1]}; {time.time() - t_train:.0f}s)", flush=True)
            train_s = time.time() - t_train
            model.load_state_dict(best_state)
            (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            torch.save(best_state, out_dir / "checkpoints" / f"{tag}_best_model.pt")

            # Latência de inferência por frame (lote de 1) e memória de pico na inferência.
            model.eval()
            sample = pre(data["test"]["A"].reshape(-1, 128, 128)[:24])
            predict_frames(model, sample[:4], dev, batch=1, amp=amp)  # aquecimento
            if dev.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            predict_frames(model, sample[4:], dev, batch=1, amp=amp)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            lat_ms = 1000 * (time.perf_counter() - t0) / len(sample[4:])
            mem_mb = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else float("nan")

            (out_dir / "probs").mkdir(parents=True, exist_ok=True)
            for split in ("valid", "test", "visual"):
                d = data[split]
                shp = d["A"].shape
                P = predict_frames(model, pre(d["A"].reshape(-1, *shp[-2:])), dev, batch=32, amp=amp)
                np.savez_compressed(out_dir / "probs" / f"{tag}_{split}.npz", P=P.reshape(shp).astype(np.float16),
                                    regime=d["regime"], traj=d["traj"], times=d["times"])

            run = {
                "arch": arch, "arquitetura": models.ARCH_NAMES[arch], "pre": pre.describe(),
                "pretreinado": pretrained and arch != "unet", "epocas": epochs, "melhor_epoca": best[1],
                "dice_valid_melhor": best[0], "batch": bs, "lr": cfg["treino"]["lr"][arch], "seed": cfg["seed"],
                "amp": amp, "mask2former_pontos": cfg["treino"]["mask2former_pontos"] if arch == "mask2former" else None,
                "tau_gt": TAU, "n_parametros": n_params, "n_frames_treino": int(len(Xtr)),
                "historico": hist, "tempo_treino_s": train_s, "latencia_rede_ms_frame": lat_ms,
                "latencia_pre_ms_frame": pre_ms_frame, "memoria_pico_MB": mem_mb, "dispositivo": str(dev),
                "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else platform.processor(),
                "versoes": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
                "piloto": args.piloto,
            }
            run_path.parent.mkdir(parents=True, exist_ok=True)
            run_path.write_text(json.dumps(run, indent=1, ensure_ascii=False))
            print(f"[{tag}] ok: {train_s / 60:.1f} min, melhor época {best[1]} (Dice valid {best[0]:.4f}), "
                  f"{n_params / 1e6:.2f} M parâmetros, {lat_ms:.1f} ms/frame", flush=True)
            del model, opt, best_state
            if dev.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
