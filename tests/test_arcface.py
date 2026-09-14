import torch

from src.config import STgramMFNConfig
from src.modules.arcface import ArcMarginProduct


def make_cfg(**overrides) -> STgramMFNConfig:
    base = dict(num_classes=5, use_arcface=True, arcface_m=0.7, arcface_s=30.0)
    base.update(overrides)
    return STgramMFNConfig(**base)


def test_output_shape():
    cfg = make_cfg()
    head = ArcMarginProduct(cfg)
    x = torch.randn(4, cfg.embed_dim)
    labels = torch.randint(0, cfg.num_classes, (4,))
    out = head(x, labels)
    assert out.shape == (4, cfg.num_classes)


def test_margin_lowers_target_logit():
    """For the target class, ArcFace returns cos(theta + m) * s, which is below
    the plain scaled cosine cos(theta) * s whenever the margin actually applies.
    """
    cfg = make_cfg(arcface_m=0.7, arcface_s=30.0)
    head = ArcMarginProduct(cfg)
    x = torch.randn(8, cfg.embed_dim)
    labels = torch.randint(0, cfg.num_classes, (8,))
    out = head(x, labels)
    cosine = torch.nn.functional.linear(
        torch.nn.functional.normalize(x), torch.nn.functional.normalize(head.weight)
    )
    target_plain = cosine.gather(1, labels.view(-1, 1)).squeeze(1) * cfg.arcface_s
    target_margin = out.gather(1, labels.view(-1, 1)).squeeze(1)
    assert (target_margin <= target_plain + 1e-5).all()


def test_sub_centers_reduce_to_class_logits():
    cfg = make_cfg(arcface_sub=3)
    head = ArcMarginProduct(cfg)
    assert head.weight.shape == (cfg.num_classes * 3, cfg.embed_dim)
    x = torch.randn(4, cfg.embed_dim)
    labels = torch.randint(0, cfg.num_classes, (4,))
    out = head(x, labels)
    assert out.shape == (4, cfg.num_classes)


def test_no_nan_on_large_logits():
    cfg = make_cfg()
    head = ArcMarginProduct(cfg)
    x = torch.randn(16, cfg.embed_dim) * 100
    labels = torch.randint(0, cfg.num_classes, (16,))
    out = head(x, labels)
    assert torch.isfinite(out).all()
