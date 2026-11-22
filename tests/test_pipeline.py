import numpy as np
import pytest
import torch

from src import models, postprocessing, preprocessing
from src.data.experiment import FrameDataset, predict_frames
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
    assert postprocessing.apply("Q4", P, P, area_min=32).sum() == 36
    assert postprocessing.apply("Q4", P, P, area_min=37).sum() == 0


def test_rms():
    p = preprocessing.build("P2").fit(A3)
    np.testing.assert_allclose(np.sqrt(np.mean(p(A3) ** 2)), 1, rtol=1e-4)


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
def test_modelos_forma_e_perda(arch):
    torch.manual_seed(0)
    m = models.build(arch, pretrained=False)
    x = torch.rand(2, 1, 128, 128)
    y = (torch.rand(2, 128, 128) > 0.7).float()
    loss = m.loss(x, y)
    loss.backward()
    assert torch.isfinite(loss)
    m.eval()
    with torch.no_grad():
        p = m.predict_proba(x)
    assert p.shape == (2, 128, 128) and float(p.min()) >= 0 and float(p.max()) <= 1


def test_unet_periodica():
    """U-Net com padding circular comuta com translações por múltiplos de 16 px (4 poolings)."""
    torch.manual_seed(0)
    m = models.build("unet").eval()
    x = torch.rand(1, 1, 128, 128)
    with torch.no_grad():
        a = m.predict_proba(torch.roll(x, (16, 32), (2, 3)))
        b = torch.roll(m.predict_proba(x), (16, 32), (1, 2))
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_dataset_e_inferencia():
    M = (A3 > 0.5).astype(np.uint8)
    ds = FrameDataset(A3, M, train=True)
    x, y = ds[0]
    assert x.shape == (1, 128, 128) and y.shape == (128, 128)
    # o rótulo acompanha o pixel após rotação/reflexão/translação
    np.testing.assert_array_equal((x[0] > 0.5).numpy(), y.numpy().astype(bool))

    class Limiar(torch.nn.Module):
        def predict_proba(self, x):
            return torch.sigmoid((x[:, 0] - 0.5) * 100)
    P = predict_frames(Limiar(), A3, "cpu", batch=2)
    np.testing.assert_array_equal(P > 0.5, A3 > 0.5)
