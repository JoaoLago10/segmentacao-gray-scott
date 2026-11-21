import numpy as np
import pytest
import torch

from src import models, postprocessing, preprocessing
from src.data.experiment import BlockDataset, periodic_crop, predict_frames
from src.evaluation.metrics import bootstrap_ci, confusion, scores, temporal

rng = np.random.default_rng(0)
A = (0.5 + 0.3 * np.sin(np.linspace(0, 8 * np.pi, 128))[None, :] *
     np.cos(np.linspace(0, 6 * np.pi, 128))[:, None]).astype(np.float32)
A3 = np.stack([A, np.roll(A, 3, 1), np.roll(A, 6, 1)]) + 0.01 * rng.standard_normal((3, 128, 128)).astype(np.float32)


@pytest.mark.parametrize("pid", preprocessing.PRE_IDS)
def test_pre_forma_e_finitos(pid):
    p = preprocessing.build(pid).fit(A3)
    y = p(A3)
    assert y.shape == A3.shape and y.dtype == np.float32 and np.isfinite(y).all()


@pytest.mark.parametrize("pid", ["P6", "P7", "P8", "P9", "P10"])
def test_pre_espacial_periodico(pid):
    """Filtro periódico comuta com translação circular."""
    p = preprocessing.build(pid).fit(A3)
    a = A3[:1]
    s = (5, 7) if pid != "P10" else (4, 8)  # wavelet com 2 níveis: invariância só a deslocamentos múltiplos de 4
    np.testing.assert_allclose(p(np.roll(a, s, (1, 2))), np.roll(p(a), s, (1, 2)), atol=1e-4)


def test_pre_ajuste_so_no_treino():
    p = preprocessing.build("P1").fit(A3)
    np.testing.assert_allclose(p(A3).mean(), 0, atol=1e-5)
    assert abs(p(A3 + 1).mean() - 1 / A3.std()) < 1e-3


@pytest.mark.parametrize("qid", postprocessing.POS_IDS)
def test_pos_binario(qid):
    P = rng.random((4, 128, 128)).astype(np.float32)
    for params in postprocessing.grid(qid)[:2]:
        M = postprocessing.apply(qid, P, A3[[0, 1, 2, 0]], **params)
        assert M.shape == P.shape and set(np.unique(M)) <= {0, 1}


def test_q5_periodico():
    """Um objeto cortado pelas bordas conta como um só componente no toro."""
    P = np.zeros((1, 128, 128), np.float32)
    P[0, :3, :3] = P[0, -3:, :3] = P[0, :3, -3:] = P[0, -3:, -3:] = 1  # 36 px no toro
    assert postprocessing.apply("Q5", P, P, area_min=32).sum() == 36
    assert postprocessing.apply("Q5", P, P, area_min=37).sum() == 0


def test_q6_preenche_buraco():
    P = np.zeros((1, 128, 128), np.float32)
    P[0, 10:20, 10:20] = 1
    P[0, 14:16, 14:16] = 0
    assert postprocessing.apply("Q6", P, P, area_max=4)[0, 10:20, 10:20].all()


def test_metricas():
    g = np.zeros((2, 4, 4), np.uint8); g[:, :2] = 1
    p = np.zeros_like(g); p[:, :1] = 1; p[:, 3, 3] = 1
    c = confusion(p, g).sum(0)
    s = scores(c)
    assert s["IoU"] == pytest.approx(8 / 18) and s["Dice"] == pytest.approx(16 / 26)
    assert s["Precision"] == pytest.approx(8 / 10) and s["Recall"] == pytest.approx(0.5)
    assert scores(np.array([0, 0, 0, 16]))["IoU"] == 1.0
    m, h = bootstrap_ci(np.ones(10))
    assert m == 1 and h == 0
    t = temporal(g, g)
    assert t["ErroArea"] == 0 and t["ErroCentroide"] == 0 and t["ErroConsist"] == 0


@pytest.mark.parametrize("arch", list(models.ARCHS))
def test_modelos_forma_e_permutacao(arch):
    torch.manual_seed(0)
    m = models.build(arch).eval()
    x = torch.randn(2, 256, 3)
    with torch.no_grad():
        y = m(x)
        perm = torch.randperm(256)
        y2 = m(x[:, perm])
    assert y.shape == (2, 256)
    if arch != "pointnet2":  # FPS começa no 1º ponto em eval: o subconjunto amostrado depende da ordem
        torch.testing.assert_close(y[:, perm], y2, atol=1e-4, rtol=1e-4)


def test_bloco_e_inferencia():
    M = (A3 > 0.5).astype(np.uint8)
    ds = BlockDataset(A3, M, 64, train=True)
    pts, lab = ds[0]
    assert pts.shape == (4096, 3) and lab.shape == (4096,)
    # o rótulo acompanha o ponto: limiar no atributo reproduz a máscara do bloco
    np.testing.assert_array_equal((pts[:, 2] > 0.5).numpy(), lab.numpy().astype(bool))
    np.testing.assert_array_equal(periodic_crop(A3[0], 120, 120, 16)[:8, :8], A3[0, 120:, 120:])

    class Ident(torch.nn.Module):
        def forward(self, x):
            return (x[..., 2] - 0.5) * 100
    P = predict_frames(Ident(), A3, 64, 32, "cpu", batch=8)
    np.testing.assert_array_equal(P > 0.5, A3 > 0.5)
