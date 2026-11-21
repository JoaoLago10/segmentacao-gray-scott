"""Treina uma arquitetura de nuvem de pontos para cada pré-processamento e salva as probabilidades.

Uso:
    python scripts/05_treinar.py --arch pointnet --pre P0 P1 ... P10
    python scripts/05_treinar.py --arch all --pre all            # 33 treinos (3 x 11)
    python scripts/05_treinar.py --arch all --pre all --piloto   # versão reduzida para CPU

Saídas (``results/exp`` ou ``results/piloto``):
    runs/{arch}_{pre}.json        configuração, perdas por época, tempo, parâmetros, latência, memória
    probs/{arch}_{pre}_{split}.npz  probabilidades float16 (n_traj, n_t, H, W) de valid, test e visual
    checkpoints/{arch}_{pre}.pt
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import models, preprocessing  # noqa: E402
from src.data.experiment import ROOT, BlockDataset, load_config, load_split, predict_frames  # noqa: E402
from src.ground_truth.generation import TAU, gt_mask, valid_frames  # noqa: E402


def dice_bce(logits, y):
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    inter = (p * y).sum(1)
    dice = 1 - (2 * inter + 1) / (p.sum(1) + y.sum(1) + 1)
    return bce + dice.mean()


@torch.no_grad()
def recalibrate_bn(model, dl, dev, n_batches=100):
    """Recalcula as estatísticas do BatchNorm com os pesos finais (média cumulativa).

    Com a política one-cycle, as médias móveis do BN ficam defasadas dos pesos; sem isso,
    o modo de avaliação diverge do modo de treino (verificado no piloto: acurácia 0,71 vs 0,94).
    """
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    for i, (x, _) in enumerate(dl):
        if i >= n_batches:
            break
        model(x.to(dev))
    model.eval()


def subset_piloto(d: dict, n_traj: int, every: int) -> dict:
    """Piloto: n_traj trajetórias por regime e 1 de cada ``every`` frames."""
    keep = np.concatenate([np.nonzero(d["regime"] == r)[0][:n_traj] for r in dict.fromkeys(d["regime"])])
    out = {k: d[k][keep] for k in ("A", "B", "regime", "traj")}
    out["A"], out["B"] = out["A"][:, ::every], out["B"][:, ::every]
    out["times"] = d["times"][::every]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", nargs="+", default=["all"])
    ap.add_argument("--pre", nargs="+", default=["all"])
    ap.add_argument("--epocas", type=int, default=None)
    ap.add_argument("--piloto", action="store_true", help="dados e épocas reduzidos (validação do pipeline em CPU)")
    ap.add_argument("--amp", action="store_true", help="precisão mista (só CUDA)")
    ap.add_argument("--refazer", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    archs = list(models.ARCHS) if args.arch == ["all"] else args.arch
    pres = preprocessing.PRE_IDS if args.pre == ["all"] else args.pre
    out_dir = ROOT / ("results/piloto" if args.piloto else "results/exp")
    epochs = args.epocas or (cfg["treino"]["epocas"] if not args.piloto else 4)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, torch.get_num_threads()))

    data = {s: load_split(cfg, s) for s in ("train", "valid", "test", "visual")}
    if args.piloto:
        data["train"] = subset_piloto(data["train"], 4, 2)
        data["valid"] = subset_piloto(data["valid"], 2, 5)
        data["test"] = subset_piloto(data["test"], 3, 5)
    for d in data.values():
        d["M"] = gt_mask(d["B"])
        d["ok"] = valid_frames(d["B"], d["times"])

    tr = data["train"]
    A_fit = tr["A"][tr["ok"]]                      # (n_frames, H, W) só frames válidos
    M_fit = tr["M"][tr["ok"]]
    size, stride = cfg["nuvem"]["bloco"], cfg["nuvem"]["passo_inferencia"]
    if args.piloto:
        stride = size  # piloto: 4 blocos sem sobreposição por frame

    for arch in archs:
        for pid in pres:
            tag = f"{arch}_{pid}"
            run_path = out_dir / "runs" / f"{tag}.json"
            if run_path.exists() and not args.refazer:
                print(f"[{tag}] já existe, pulando", flush=True)
                continue
            torch.manual_seed(cfg["seed"])
            np.random.seed(cfg["seed"])
            pre = preprocessing.build(pid).fit(A_fit)
            t0 = time.time()
            Xtr = pre(A_fit)
            pre_s = time.time() - t0
            pre_ms_frame = 1000 * pre_s / len(Xtr)

            model = models.build(arch).to(dev)
            n_params = sum(p.numel() for p in model.parameters())
            bs = cfg["treino"]["batch"][arch]
            dl = torch.utils.data.DataLoader(
                BlockDataset(Xtr, M_fit, size, train=True), batch_size=bs, shuffle=True,
                num_workers=cfg["treino"]["workers"], drop_last=True,
                generator=torch.Generator().manual_seed(cfg["seed"]))
            va = data["valid"]
            Xva = pre(va["A"][va["ok"]])
            Mva = va["M"][va["ok"]]
            sel = np.linspace(0, len(Xva) - 1, min(256, len(Xva))).astype(int)
            dl_va = torch.utils.data.DataLoader(BlockDataset(Xva[sel], Mva[sel], size, train=False),
                                                batch_size=bs)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["treino"]["lr"],
                                    weight_decay=cfg["treino"]["weight_decay"])
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["treino"]["lr"],
                                                        total_steps=epochs * len(dl))
            scaler = torch.amp.GradScaler("cuda", enabled=args.amp and dev.type == "cuda")
            hist = []
            t_train = time.time()
            for ep in range(epochs):
                model.train()
                tot, n = 0.0, 0
                for x, y in dl:
                    x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                    with torch.autocast(dev.type, enabled=args.amp and dev.type == "cuda"):
                        loss = dice_bce(model(x).float(), y)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    tot, n = tot + loss.item() * len(x), n + len(x)
                if ep == epochs - 1:
                    recalibrate_bn(model, dl, dev)
                model.eval()
                vtot, vn = 0.0, 0
                with torch.no_grad():
                    for x, y in dl_va:
                        x, y = x.to(dev), y.to(dev)
                        vtot += dice_bce(model(x).float(), y).item() * len(x)
                        vn += len(x)
                hist.append({"epoca": ep + 1, "perda_treino": tot / n, "perda_valid": vtot / vn,
                             "tempo_s": time.time() - t_train})
                print(f"[{tag}] época {ep + 1}/{epochs} treino {tot / n:.4f} valid {vtot / vn:.4f} "
                      f"({time.time() - t_train:.0f}s)", flush=True)
            train_s = time.time() - t_train

            (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_dir / "checkpoints" / f"{tag}.pt")

            # Latência por frame (rede, 1 frame = todos os blocos dele num lote) e memória de pico.
            model.eval()
            te = data["test"]
            sample = pre(te["A"].reshape(-1, *te["A"].shape[-2:])[:24])
            n_blocks = (te["A"].shape[-1] // stride) ** 2
            predict_frames(model, sample[:2], size, stride, dev, batch=n_blocks)  # aquecimento
            if dev.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for f in sample[2:]:
                predict_frames(model, f[None], size, stride, dev, batch=n_blocks)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            lat_ms = 1000 * (time.perf_counter() - t0) / len(sample[2:])
            mem_mb = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else float("nan")

            (out_dir / "probs").mkdir(parents=True, exist_ok=True)
            for split in ("valid", "test", "visual"):
                d = data[split]
                shp = d["A"].shape
                X = pre(d["A"].reshape(-1, *shp[-2:]))
                P = predict_frames(model, X, size, stride, dev, batch=64).reshape(shp)
                np.savez_compressed(out_dir / "probs" / f"{tag}_{split}.npz", P=P.astype(np.float16),
                                    regime=d["regime"], traj=d["traj"], times=d["times"])

            run = {
                "arch": arch, "arquitetura": models.ARCH_NAMES[arch], "pre": pre.describe(),
                "epocas": epochs, "batch": bs, "lr": cfg["treino"]["lr"], "seed": cfg["seed"],
                "tau_gt": TAU, "bloco": size, "passo_inferencia": stride, "n_parametros": n_params,
                "n_frames_treino": int(len(Xtr)), "historico": hist, "tempo_treino_s": train_s,
                "latencia_rede_ms_frame": lat_ms, "latencia_pre_ms_frame": pre_ms_frame,
                "memoria_pico_MB": mem_mb, "dispositivo": str(dev),
                "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else platform.processor(),
                "versoes": {"python": platform.python_version(), "torch": torch.__version__,
                            "numpy": np.__version__},
                "piloto": args.piloto,
            }
            run_path.parent.mkdir(parents=True, exist_ok=True)
            run_path.write_text(json.dumps(run, indent=1, ensure_ascii=False))
            print(f"[{tag}] ok: {train_s / 60:.1f} min, {n_params / 1e6:.2f} M parâmetros, "
                  f"{lat_ms:.1f} ms/frame", flush=True)


if __name__ == "__main__":
    main()
