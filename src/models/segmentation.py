"""U-Net, SegFormer e Mask2Former com interface comum (protocolo da turma, Seção 7).

Interface de todas as redes:
    model.loss(x, y)        -> perda escalar de treino;  x (B, 1, H, W), y (B, H, W) em {0, 1}
    model.predict_proba(x)  -> P(padrão) por pixel (B, H, W) na resolução da entrada

Adaptações documentadas:
- Entrada: 1 canal (campo A pré-processado). Nas redes pré-treinadas, o canal é replicado
  em 3 para reaproveitar os pesos da primeira camada.
- Sem upscaling da entrada (protocolo, Seção 3): as redes recebem 128 x 128. SegFormer e
  Mask2Former predizem em 1/4 da resolução (32 x 32); só a **saída** é interpolada
  (bilinear) para 128 x 128.
- U-Net com padding circular (contorno periódico do Gray-Scott). Os Transformers usam o
  padding padrão; a periodicidade entra por translações circulares no aumento de dados.
- Perda: BCE + Dice (U-Net e SegFormer); a perda nativa de Mask2Former (Hungarian matching,
  CE de classe + BCE + Dice de máscara) com duas classes semânticas (fundo e padrão).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SEGFORMER_CKPT = "nvidia/mit-b0"
MASK2FORMER_CKPT = "facebook/mask2former-swin-tiny-ade-semantic"


def bce_dice(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits).flatten(1)
    t = y.flatten(1)
    dice = 1 - (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)
    return bce + dice.mean()


# --------------------------------------------------------------------- U-Net
class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, padding_mode="circular", bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, padding_mode="circular", bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """U-Net (Ronneberger et al., 2015): 4 níveis (128 -> 8 px), base de 32 canais."""

    def __init__(self, in_ch=1, base=32):
        super().__init__()
        ch = [base * 2 ** i for i in range(5)]
        self.downs = nn.ModuleList([DoubleConv(in_ch, ch[0])] +
                                   [DoubleConv(ch[i], ch[i + 1]) for i in range(4)])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([DoubleConv(2 * ch[i], ch[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(ch[0], 1, 1)

    def logits(self, x):
        skips = []
        for i, d in enumerate(self.downs):
            x = d(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.ups, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        return self.head(x).squeeze(1)

    def loss(self, x, y):
        return bce_dice(self.logits(x), y)

    def predict_proba(self, x):
        return torch.sigmoid(self.logits(x))


# ----------------------------------------------------------------- SegFormer
class SegFormer(nn.Module):
    """SegFormer-B0 (Xie et al., 2021): encoder MiT hierárquico + decoder MLP leve."""

    def __init__(self, pretrained=True):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        if pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_CKPT, num_labels=1)
        else:
            self.net = SegformerForSemanticSegmentation(SegformerConfig.from_pretrained(SEGFORMER_CKPT, num_labels=1))

    def logits(self, x):
        out = self.net(pixel_values=x.expand(-1, 3, -1, -1)).logits          # (B, 1, H/4, W/4)
        return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

    def loss(self, x, y):
        return bce_dice(self.logits(x), y)

    def predict_proba(self, x):
        return torch.sigmoid(self.logits(x))


# --------------------------------------------------------------- Mask2Former
class Mask2Former(nn.Module):
    """Mask2Former (Cheng et al., 2022) com backbone Swin-T, em modo semântico de 2 classes."""

    def __init__(self, pretrained=True, train_num_points=12544):
        super().__init__()
        from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation
        kw = dict(num_labels=2, id2label={0: "fundo", 1: "padrao"}, label2id={"fundo": 0, "padrao": 1},
                  train_num_points=train_num_points)
        if pretrained:
            self.net = Mask2FormerForUniversalSegmentation.from_pretrained(
                MASK2FORMER_CKPT, ignore_mismatched_sizes=True, **kw)
        else:
            self.net = Mask2FormerForUniversalSegmentation(Mask2FormerConfig.from_pretrained(MASK2FORMER_CKPT, **kw))

    def loss(self, x, y):
        mask_labels, class_labels = [], []
        for t in y:
            present = [c for c in (0, 1) if ((t == c).any())]
            mask_labels.append(torch.stack([(t == c).float() for c in present]))
            class_labels.append(torch.tensor(present, device=x.device))
        out = self.net(pixel_values=x.expand(-1, 3, -1, -1), mask_labels=mask_labels, class_labels=class_labels)
        return out.loss

    def predict_proba(self, x):
        out = self.net(pixel_values=x.expand(-1, 3, -1, -1))
        cls = out.class_queries_logits.softmax(-1)[..., :-1]                   # (B, Q, 2), sem "vazio"
        masks = out.masks_queries_logits.sigmoid()                             # (B, Q, H/4, W/4)
        seg = torch.einsum("bqc,bqhw->bchw", cls, masks)
        seg = F.interpolate(seg, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return seg[:, 1] / seg.sum(1).clamp_min(1e-8)


ARCHS = {"unet": UNet, "segformer": SegFormer, "mask2former": Mask2Former}
ARCH_NAMES = {"unet": "U-Net", "segformer": "SegFormer", "mask2former": "Mask2Former"}


def build(name: str, pretrained: bool = True, mask2former_points: int = 12544) -> nn.Module:
    if name == "unet":
        return UNet()
    if name == "mask2former":
        return Mask2Former(pretrained=pretrained, train_num_points=mask2former_points)
    return SegFormer(pretrained=pretrained)
