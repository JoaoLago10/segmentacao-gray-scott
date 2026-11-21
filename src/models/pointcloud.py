"""Arquiteturas de segmentação de nuvens de pontos (PyTorch puro).

Cada frame do Gray-Scott vira uma nuvem de pontos: cada pixel é um ponto com
coordenadas (x, y) e atributo a = P(A)(x, y) (o campo A pré-processado). Assim a
entrada tem forma (B, N, 3) = [x, y, a] e a saída são logits por ponto (B, N).

- PointNet (Qi et al., CVPR 2017): MLP compartilhado + max-pooling global.
  Versão "vanilla", sem T-Nets: a nuvem é 2D com orientação canônica e as rotações de
  90° já entram como aumento de dados.
- PointNet++ SSG (Qi et al., NeurIPS 2017): set abstraction hierárquico com amostragem
  por ponto mais distante (FPS) e agrupamento por k vizinhos (em vez de ball query), mais
  propagação de características por interpolação inversa à distância (3-NN).
- DGCNN (Wang et al., ACM TOG 2019): EdgeConv com grafo k-NN dinâmico. O 1º grafo usa
  as coordenadas espaciais (x, y); os seguintes usam o espaço de características.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp1d(channels, last_act=True):
    layers = []
    for i in range(len(channels) - 1):
        layers.append(nn.Conv1d(channels[i], channels[i + 1], 1, bias=False))
        layers.append(nn.BatchNorm1d(channels[i + 1]))
        if last_act or i < len(channels) - 2:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def mlp2d(channels):
    layers = []
    for i in range(len(channels) - 1):
        layers += [nn.Conv2d(channels[i], channels[i + 1], 1, bias=False),
                   nn.BatchNorm2d(channels[i + 1]), nn.LeakyReLU(0.2, inplace=True)]
    return nn.Sequential(*layers)


def knn(query: torch.Tensor, ref: torch.Tensor, k: int, chunk: int = 2048) -> torch.Tensor:
    """Índices (B, M, k) dos k vizinhos de ``query`` (B, M, D) em ``ref`` (B, N, D)."""
    out = []
    for s in range(0, query.shape[1], chunk):
        d = torch.cdist(query[:, s:s + chunk], ref)
        out.append(d.topk(k, dim=-1, largest=False).indices)
    return torch.cat(out, 1)


def gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x (B, N, C), idx (B, ...) -> (B, ..., C)."""
    B = x.shape[0]
    flat = idx.reshape(B, -1)
    out = torch.gather(x, 1, flat.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    return out.reshape(*idx.shape, x.shape[-1])


def fps(xyz: torch.Tensor, m: int) -> torch.Tensor:
    """Amostragem por ponto mais distante: índices (B, m). Ponto inicial aleatório (determinístico em eval)."""
    B, N, _ = xyz.shape
    idx = torch.zeros(B, m, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), float("inf"), device=xyz.device)
    far = (torch.randint(0, N, (B,), device=xyz.device) if torch.is_grad_enabled()
           else torch.zeros(B, dtype=torch.long, device=xyz.device))
    ar = torch.arange(B, device=xyz.device)
    for i in range(m):
        idx[:, i] = far
        c = xyz[ar, far].unsqueeze(1)
        dist = torch.minimum(dist, ((xyz - c) ** 2).sum(-1))
        far = dist.argmax(-1)
    return idx


# ----------------------------------------------------------------- PointNet
class PointNetSeg(nn.Module):
    def __init__(self, in_ch=3, glob=1024):
        super().__init__()
        self.local = mlp1d([in_ch, 64, 64])
        self.glob = mlp1d([64, 128, glob])
        self.head = nn.Sequential(mlp1d([64 + glob, 512, 256, 128]), nn.Dropout(0.3),
                                  nn.Conv1d(128, 1, 1))

    def forward(self, x):  # x (B, N, C)
        f = self.local(x.transpose(1, 2))           # (B, 64, N)
        g = self.glob(f).max(-1, keepdim=True)[0]   # (B, G, 1)
        h = torch.cat([f, g.expand(-1, -1, f.shape[-1])], 1)
        return self.head(h).squeeze(1)


# ---------------------------------------------------------------- PointNet++
class SetAbstraction(nn.Module):
    def __init__(self, m_ratio, k, in_ch, channels):
        super().__init__()
        self.m_ratio, self.k = m_ratio, k
        self.mlp = mlp2d([in_ch + 2] + channels)

    def forward(self, xyz, feat):  # xyz (B, N, 2), feat (B, N, C)
        m = max(1, xyz.shape[1] // self.m_ratio)
        cidx = fps(xyz, m)
        new_xyz = gather(xyz, cidx)                          # (B, m, 2)
        nidx = knn(new_xyz, xyz, self.k)                     # (B, m, k)
        g_xyz = gather(xyz, nidx) - new_xyz.unsqueeze(2)     # coordenadas relativas
        g = torch.cat([g_xyz, gather(feat, nidx)], -1)       # (B, m, k, 2+C)
        g = self.mlp(g.permute(0, 3, 1, 2)).max(-1)[0]       # (B, C', m)
        return new_xyz, g.transpose(1, 2)


class FeaturePropagation(nn.Module):
    def __init__(self, in_ch, channels):
        super().__init__()
        self.mlp = mlp1d([in_ch] + channels)

    def forward(self, xyz1, xyz2, f1, f2):  # interpola f2 (esparso) para xyz1 (denso)
        d, idx = torch.cdist(xyz1, xyz2).topk(min(3, xyz2.shape[1]), -1, largest=False)
        w = 1.0 / (d + 1e-8)
        w = w / w.sum(-1, keepdim=True)
        interp = (gather(f2, idx) * w.unsqueeze(-1)).sum(2)
        h = interp if f1 is None else torch.cat([f1, interp], -1)
        return self.mlp(h.transpose(1, 2)).transpose(1, 2)


class PointNet2Seg(nn.Module):
    def __init__(self, in_ch=3, k=16):
        super().__init__()
        c0 = in_ch - 2
        self.sa1 = SetAbstraction(4, k, c0, [32, 32, 64])
        self.sa2 = SetAbstraction(4, k, 64, [64, 64, 128])
        self.sa3 = SetAbstraction(4, k, 128, [128, 128, 256])
        self.fp3 = FeaturePropagation(256 + 128, [256, 256])
        self.fp2 = FeaturePropagation(256 + 64, [256, 128])
        self.fp1 = FeaturePropagation(128 + c0, [128, 128, 128])
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Conv1d(128, 1, 1))

    def forward(self, x):
        xyz0, f0 = x[..., :2].contiguous(), x[..., 2:].contiguous()
        xyz1, f1 = self.sa1(xyz0, f0)
        xyz2, f2 = self.sa2(xyz1, f1)
        xyz3, f3 = self.sa3(xyz2, f2)
        f2 = self.fp3(xyz2, xyz3, f2, f3)
        f1 = self.fp2(xyz1, xyz2, f1, f2)
        f0 = self.fp1(xyz0, xyz1, f0, f1)
        return self.head(f0.transpose(1, 2)).squeeze(1)


# -------------------------------------------------------------------- DGCNN
def edge_features(feat, idx):
    """feat (B, N, C), idx (B, N, k) -> (B, 2C, N, k) com [x_j - x_i, x_i]."""
    nb = gather(feat, idx)
    center = feat.unsqueeze(2).expand_as(nb)
    return torch.cat([nb - center, center], -1).permute(0, 3, 1, 2)


class DGCNNSeg(nn.Module):
    def __init__(self, in_ch=3, k=16, emb=1024):
        super().__init__()
        self.k = k
        self.ec1 = mlp2d([2 * in_ch, 64, 64])
        self.ec2 = mlp2d([2 * 64, 64, 64])
        self.ec3 = mlp2d([2 * 64, 64])
        self.emb = mlp1d([192, emb])
        self.head = nn.Sequential(mlp1d([192 + emb, 256, 256, 128]), nn.Dropout(0.3),
                                  nn.Conv1d(128, 1, 1))

    def forward(self, x):
        idx = knn(x[..., :2], x[..., :2], self.k)            # grafo espacial
        x1 = self.ec1(edge_features(x, idx)).max(-1)[0].transpose(1, 2)
        idx = knn(x1, x1, self.k)                             # grafo dinâmico
        x2 = self.ec2(edge_features(x1, idx)).max(-1)[0].transpose(1, 2)
        idx = knn(x2, x2, self.k)
        x3 = self.ec3(edge_features(x2, idx)).max(-1)[0].transpose(1, 2)
        h = torch.cat([x1, x2, x3], -1).transpose(1, 2)       # (B, 192, N)
        g = self.emb(h).max(-1, keepdim=True)[0]
        return self.head(torch.cat([h, g.expand(-1, -1, h.shape[-1])], 1)).squeeze(1)


ARCHS = {"pointnet": PointNetSeg, "pointnet2": PointNet2Seg, "dgcnn": DGCNNSeg}
ARCH_NAMES = {"pointnet": "PointNet", "pointnet2": "PointNet++", "dgcnn": "DGCNN"}


def build(name: str) -> nn.Module:
    return ARCHS[name]()
