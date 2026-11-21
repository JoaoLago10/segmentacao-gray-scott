"""Pré-processamentos P1--P10 aplicados ao campo de entrada A.

Regras (Seções 0.5 e 2.3 do histórico):
- Uma transformação por experimento; P0 = "nenhum" (dados brutos, condição "sem pré").
- Estatísticas (P1--P5) ajustadas **só no treino** e reaplicadas sem recalibração na
  validação e no teste.
- Filtros espaciais (P6--P10) respeitam o contorno periódico (modo ``wrap``).
- O ground truth é sempre gerado do B bruto; o pré-processamento nunca o altera.

Todas as funções recebem arrays (..., H, W) em float32 e devolvem a mesma forma.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pywt
from scipy import ndimage
from skimage.restoration import denoise_bilateral


@dataclass
class Pre:
    id: str
    nome: str
    params: dict = field(default_factory=dict)

    def fit(self, train: np.ndarray) -> "Pre":
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.transform(np.asarray(x, dtype=np.float32)).astype(np.float32)

    def describe(self) -> dict:
        return {"id": self.id, "nome": self.nome,
                "params": {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                           for k, v in self.params.items()}}


def _frames(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1, *x.shape[-2:])


# ----------------------------------------------------------------- numéricos
class Nenhum(Pre):
    def transform(self, x):
        return x


class ZScore(Pre):
    def fit(self, train):
        self.params = {"media": float(train.mean()), "desvio": float(train.std())}
        return self

    def transform(self, x):
        return (x - self.params["media"]) / self.params["desvio"]


class MinMax(Pre):
    def fit(self, train):
        self.params = {"min": float(train.min()), "max": float(train.max())}
        return self

    def transform(self, x):
        p = self.params
        return (x - p["min"]) / (p["max"] - p["min"])


class Robusta(Pre):
    def fit(self, train):
        q1, med, q3 = np.percentile(train, [25, 50, 75])
        self.params = {"mediana": float(med), "iqr": float(q3 - q1)}
        return self

    def transform(self, x):
        return (x - self.params["mediana"]) / self.params["iqr"]


class Percentis(Pre):
    """Recorte nos percentis 1 e 99 do treino (sem reescala, para isolar o efeito do recorte)."""

    def __init__(self, *a, inf=1.0, sup=99.0, **k):
        super().__init__(*a, **k)
        self.inf, self.sup = inf, sup

    def fit(self, train):
        lo, hi = np.percentile(train, [self.inf, self.sup])
        self.params = {"p_inf": self.inf, "p_sup": self.sup, "lo": float(lo), "hi": float(hi)}
        return self

    def transform(self, x):
        return np.clip(x, self.params["lo"], self.params["hi"])


class Log(Pre):
    """y = log(max(A, 0) + eps). A > 0 em t > 0 (mín. 0,09), mas o eps explícito cobre zeros."""

    def __init__(self, *a, eps=1e-3, **k):
        super().__init__(*a, **k)
        self.params = {"eps": eps}

    def transform(self, x):
        return np.log(np.maximum(x, 0.0) + self.params["eps"])


# ----------------------------------------------------------------- espaciais
class Gaussiano(Pre):
    def __init__(self, *a, sigma=1.0, **k):
        super().__init__(*a, **k)
        self.params = {"sigma_px": sigma, "modo": "wrap"}

    def transform(self, x):
        s = self.params["sigma_px"]
        return ndimage.gaussian_filter(x, sigma=(0,) * (x.ndim - 2) + (s, s), mode="wrap")


class Mediana(Pre):
    def __init__(self, *a, size=3, **k):
        super().__init__(*a, **k)
        self.params = {"janela": size, "modo": "wrap"}

    def transform(self, x):
        s = self.params["janela"]
        return ndimage.median_filter(x, size=(1,) * (x.ndim - 2) + (s, s), mode="wrap")


class Bilateral(Pre):
    def __init__(self, *a, sigma_color=0.05, sigma_spatial=1.0, win=5, **k):
        super().__init__(*a, **k)
        self.params = {"sigma_cor": sigma_color, "sigma_espacial_px": sigma_spatial,
                       "janela": win, "modo": "wrap"}

    def transform(self, x):
        p = self.params
        out = np.empty_like(x)
        fo, fi = _frames(out), _frames(x)
        for i in range(fi.shape[0]):
            fo[i] = denoise_bilateral(fi[i].astype(np.float64), win_size=p["janela"],
                                      sigma_color=p["sigma_cor"],
                                      sigma_spatial=p["sigma_espacial_px"], mode="wrap")
        return out


class PeronaMalik(Pre):
    """Difusão anisotrópica de Perona e Malik (1990), g(s) = exp(-(s/kappa)^2), vizinhança-4 periódica."""

    def __init__(self, *a, niter=10, kappa=0.05, gamma=0.2, **k):
        super().__init__(*a, **k)
        self.params = {"iteracoes": niter, "kappa": kappa, "gamma": gamma, "modo": "wrap"}

    def transform(self, x):
        p = self.params
        u = x.astype(np.float32).copy()
        ax_y, ax_x = x.ndim - 2, x.ndim - 1
        for _ in range(p["iteracoes"]):
            flux = 0.0
            for ax in (ax_y, ax_x):
                for s in (1, -1):
                    d = np.roll(u, -s, axis=ax) - u
                    flux = flux + np.exp(-(d / p["kappa"]) ** 2) * d
            u = u + p["gamma"] * flux
        return u


class Wavelet(Pre):
    """Limiarização suave BayesShrink (Chang et al., 2000) com extensão periódica (pywt 'periodization')."""

    def __init__(self, *a, wavelet="db2", level=2, **k):
        super().__init__(*a, **k)
        self.params = {"wavelet": wavelet, "niveis": level, "regra": "BayesShrink suave",
                       "modo": "periodization"}

    def _one(self, img):
        p = self.params
        c = pywt.wavedec2(img, p["wavelet"], mode="periodization", level=p["niveis"])
        sigma = np.median(np.abs(c[-1][-1])) / 0.6745
        out = [c[0]]
        for det in c[1:]:
            nd = []
            for d in det:
                sx = np.sqrt(max(d.var() - sigma ** 2, 0.0))
                thr = np.abs(d).max() if sx == 0 else sigma ** 2 / sx
                nd.append(pywt.threshold(d, thr, mode="soft"))
            out.append(tuple(nd))
        return pywt.waverec2(out, p["wavelet"], mode="periodization")

    def transform(self, x):
        out = np.empty_like(x)
        fo, fi = _frames(out), _frames(x)
        for i in range(fi.shape[0]):
            fo[i] = self._one(fi[i].astype(np.float64))
        return out


def build(pid: str) -> Pre:
    table = {
        "P0": (Nenhum, "Nenhum (dados brutos)"),
        "P1": (ZScore, "Normalização Z-score"),
        "P2": (MinMax, "Normalização Min-Max"),
        "P3": (Robusta, "Normalização robusta (mediana e IQR)"),
        "P4": (Percentis, "Recorte por percentis"),
        "P5": (Log, "Transformação logarítmica"),
        "P6": (Gaussiano, "Filtro gaussiano"),
        "P7": (Mediana, "Filtro de mediana"),
        "P8": (Bilateral, "Filtro bilateral"),
        "P9": (PeronaMalik, "Difusão anisotrópica"),
        "P10": (Wavelet, "Redução de ruído por wavelets"),
    }
    cls, nome = table[pid]
    return cls(pid, nome)


PRE_IDS = ["P0"] + [f"P{i}" for i in range(1, 11)]
