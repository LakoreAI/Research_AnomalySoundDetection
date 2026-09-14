import torch

from src.modules.edge import STgramMFNScorer
from src.modules.loss import anomaly_score
from src.modules.model import STgramMFN
from tests.test_model import N_FRAMES, SECS, SR, make_cfg


def test_scorer_matches_anomaly_score():
    cfg = make_cfg(use_arcface=True)
    model = STgramMFN(cfg).eval()
    scorer = STgramMFNScorer(model).eval()

    x_wav = torch.randn(4, int(SECS * SR))
    x_mel = torch.randn(4, cfg.n_mels, N_FRAMES)
    labels = torch.randint(0, cfg.num_classes, (4,))

    with torch.no_grad():
        logits, _ = model(x_wav, x_mel, labels)
        expected = anomaly_score(logits, labels)
        feature, score = scorer(x_wav, x_mel, labels)

    assert feature.shape == (4, cfg.embed_dim)
    assert score.shape == (4,)
    assert torch.allclose(score, expected, atol=1e-5)
