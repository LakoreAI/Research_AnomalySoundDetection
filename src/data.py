"""Dataset for MIMII / DCASE Task 2 self-supervised machine-ID classification.

One sample is `(x_wav, x_mel, label)`: a fixed-length raw waveform, its log-mel
spectrogram, and the machine-ID class label. Training uses only the normal
recordings of each machine (one class per physical machine); the same dataset
class serves test-time scoring, where the label is the sample's own machine ID
and the anomaly score comes from `src.modules.loss.anomaly_score`.
"""

from pathlib import Path
from typing import Dict, List, Optional

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from src.config import STgramMFNConfig
from src.modules.frontend import FeatureExtractor
from src.utils.audio_utils import machine_id_of_file, machine_of_file


def load_waveform(path: str, target_sr: int) -> torch.Tensor:
    """Load a mono waveform resampled to `target_sr`.

    Uses `soundfile` (not `torchaudio.load`, which now requires the separate
    TorchCodec package) and `torchaudio.functional.resample` for the sample-rate
    conversion. Multi-channel input is averaged to mono.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.mean(axis=1))  # (T,)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform


class ASDDataset(Dataset):
    def __init__(
        self,
        file_list: List[str],
        meta2label: Dict[str, int],
        extractor: FeatureExtractor,
        load_in_memory: bool = False,
    ):
        self.file_list = file_list
        self.meta2label = meta2label
        self.extractor = extractor
        self.load_in_memory = load_in_memory
        self.data_list = (
            [self.transform(f) for f in file_list] if load_in_memory else []
        )

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, item: int):
        if self.load_in_memory:
            return self.data_list[item]
        return self.transform(self.file_list[item])

    def transform(self, filename: str):
        """(x_wav (clip_samples,), x_mel (n_mels, n_frames), label)."""
        machine = machine_of_file(filename)
        id_str = machine_id_of_file(filename)
        label = self.meta2label[f"{machine}-{id_str}"]
        waveform = load_waveform(filename, self.extractor.cfg.sample_rate)
        x_wav, x_mel = self.extractor(waveform)
        return x_wav, x_mel, label


def make_extractor(cfg: STgramMFNConfig) -> FeatureExtractor:
    return FeatureExtractor(cfg)


def dataset_from_files(
    file_list: List[str],
    meta2label: Dict[str, int],
    cfg: STgramMFNConfig,
    load_in_memory: bool = False,
    extractor: Optional[FeatureExtractor] = None,
) -> ASDDataset:
    """Convenience builder sharing one FeatureExtractor across callers (the
    extractor holds no state, so a single instance is safe to reuse).
    """
    extractor = extractor or make_extractor(cfg)
    return ASDDataset(file_list, meta2label, extractor, load_in_memory=load_in_memory)


def resolve_dirs(base: str, names: List[str], subdir: str) -> List[str]:
    """Expand `base/<name>/<subdir>` for every name, e.g. per-machine
    train/test roots from a config list of machine names.
    """
    base_path = Path(base)
    return [str(base_path / name / subdir) for name in names]
