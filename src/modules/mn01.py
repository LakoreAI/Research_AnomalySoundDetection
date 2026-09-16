"""mn01 — the EfficientAT AudioSet embedder used as the STgram swap.

`mn01` is EfficientAT's MobileNetV3 at `width_mult=0.1` (see
github.com/fschmid56/EfficientAT, `helpers/utils.py:NAME_TO_WIDTH`). The
AudioSet checkpoint is `mn01_as_mAP_298.pt` (`mn01_im.pt` is ImageNet-only and
is not used). It has 123,783 params and returns a 96-d pooled feature.

This is a faithful re-expression of EfficientAT's `MN` + `AugmentMelSTFT`,
mirroring how `src/modules/{mobilefacenet,arcface}.py` re-express the
STgram-MFN reference: torchvision's `ConvNormActivation` is reimplemented here
so the module import graph stays torch/torchaudio-only, and module names/indices
are kept identical so the released `mn01_as` `state_dict` loads with
`strict=True`.

It is NOT interchangeable with STgram's frontend: the weights were trained on
32 kHz audio through EfficientAT's own mel transform. MIMII is 16 kHz, so the
mn01 row must resample up before calling this.
"""

import math
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch import Tensor

from src.config import MN01Config


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """Round `v` to a multiple of `divisor` (never more than 10% down)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def cnn_out_size(
    in_size: int, padding: int, dilation: int, kernel: int, stride: int
) -> int:
    s = in_size + 2 * padding - dilation * (kernel - 1) - 1
    return math.floor(s / stride + 1)


class ConvNormActivation(nn.Sequential):
    """torchvision's `ConvNormActivation`, reimplemented so the repo does not
    depend on torchvision. `[conv, norm, act]` at indices 0/1/2 keeps the
    EfficientAT checkpoint's `*.0.weight` / `*.1.weight` keys valid.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dilation: int = 1,
        inplace: bool = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., nn.Module] = nn.Conv2d,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
        layers: List[nn.Module] = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=inplace))
        super().__init__(*layers)
        self.out_channels = out_channels


class SqueezeExcitation(nn.Module):
    """Squeeze-and-excitation over one of {channel, freq, time}. For EfficientAT
    the default is channel-only (`se_dims='c'`)."""

    def __init__(
        self,
        input_dim: int,
        squeeze_dim: int,
        se_dim: int,
        activation: Callable[..., nn.Module] = nn.ReLU,
        scale_activation: Callable[..., nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, squeeze_dim)
        self.fc2 = nn.Linear(squeeze_dim, input_dim)
        if se_dim not in (1, 2, 3):
            raise ValueError(f"se_dim must be 1/2/3, got {se_dim}")
        self.se_dim = [1, 2, 3]
        self.se_dim.remove(se_dim)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, x: Tensor) -> Tensor:
        scale = torch.mean(x, self.se_dim, keepdim=True)
        shape = scale.size()
        scale = self.fc1(scale.squeeze(2).squeeze(2))
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale).view(shape)

    def forward(self, x: Tensor) -> Tensor:
        return self._scale(x) * x


class ConcurrentSEBlock(nn.Module):
    """Fuses SE layers applied concurrently on several axes (reference)."""

    def __init__(self, c_dim: int, f_dim: int, t_dim: int, se_cnf: Dict) -> None:
        super().__init__()
        dims = [c_dim, f_dim, t_dim]
        self.conc_se_layers = nn.ModuleList()
        for d in se_cnf["se_dims"]:
            input_dim = dims[d - 1]
            squeeze_dim = make_divisible(input_dim // se_cnf["se_r"], 8)
            self.conc_se_layers.append(SqueezeExcitation(input_dim, squeeze_dim, d))
        self.agg = se_cnf["se_agg"]
        if self.agg == "max":
            self.agg_op = lambda x: torch.max(x, dim=0)[0]  # noqa: E731
        elif self.agg == "avg":
            self.agg_op = lambda x: torch.mean(x, dim=0)  # noqa: E731
        elif self.agg == "add":
            self.agg_op = lambda x: torch.sum(x, dim=0)  # noqa: E731
        elif self.agg == "min":
            self.agg_op = lambda x: torch.min(x, dim=0)[0]  # noqa: E731
        else:
            raise NotImplementedError(f"SE aggregation '{self.agg}' not implemented")

    def forward(self, x: Tensor) -> Tensor:
        se_outs = [layer(x) for layer in self.conc_se_layers]
        return self.agg_op(torch.stack(se_outs, dim=0))


class InvertedResidualConfig:
    """MobileNetV3 block spec, channels adjusted by `width_mult`."""

    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation
        self.f_dim: Optional[int] = None
        self.t_dim: Optional[int] = None

    @staticmethod
    def adjust_channels(channels: int, width_mult: float) -> int:
        return make_divisible(channels * width_mult, 8)

    def out_size(self, in_size: int) -> int:
        padding = (self.kernel - 1) // 2 * self.dilation
        return cnn_out_size(in_size, padding, self.dilation, self.kernel, self.stride)


class InvertedResidual(nn.Module):
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        se_cnf: Dict,
        norm_layer: Callable[..., nn.Module],
        depthwise_norm_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__()
        if not 1 <= cnf.stride <= 2:
            raise ValueError("illegal stride value")
        self.use_res_connect = (
            cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        )
        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            ConvNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=depthwise_norm_layer,
                activation_layer=activation_layer,
            )
        )
        if cnf.use_se and se_cnf["se_dims"] is not None:
            layers.append(
                ConcurrentSEBlock(cnf.expanded_channels, cnf.f_dim, cnf.t_dim, se_cnf)
            )
        layers.append(
            ConvNormActivation(
                cnf.expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        )
        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result = result + x
        return result


def _mobilenet_v3_conf(
    width_mult: float = 1.0,
    reduced_tail: bool = False,
    dilated: bool = False,
    strides: Tuple[int, int, int, int] = (2, 2, 2, 2),
) -> Tuple[List[InvertedResidualConfig], int]:
    """EfficientAT's MobileNetV3 block table (Tables 1/2 of the paper)."""
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1
    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    setting = [
        bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
        bneck_conf(16, 3, 64, 24, False, "RE", strides[0], 1),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
        bneck_conf(24, 5, 72, 40, True, "RE", strides[1], 1),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
        bneck_conf(40, 3, 240, 80, False, "HS", strides[2], 1),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
        bneck_conf(
            112, 5, 672, 160 // reduce_divider, True, "HS", strides[3], dilation
        ),  # C4
        bneck_conf(
            160 // reduce_divider,
            5,
            960 // reduce_divider,
            160 // reduce_divider,
            True,
            "HS",
            1,
            dilation,
        ),
        bneck_conf(
            160 // reduce_divider,
            5,
            960 // reduce_divider,
            160 // reduce_divider,
            True,
            "HS",
            1,
            dilation,
        ),
    ]
    last_channel = InvertedResidualConfig.adjust_channels(
        1280 // reduce_divider, width_mult
    )
    return setting, last_channel


class MobileNetV3(nn.Module):
    """EfficientAT's `MN`. `forward` returns `(logits, features)`; `features`
    is the global-average-pooled final conv map (96-d for mn01)."""

    def __init__(
        self,
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int,
        num_classes: int = 1000,
        block: Callable[..., nn.Module] = InvertedResidual,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
        in_conv_kernel: int = 3,
        in_conv_stride: int = 2,
        in_channels: int = 1,
        head_type: str = "mlp",
        input_dims: Tuple[int, int] = (128, 1000),
        se_conf: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        depthwise_norm_layer = norm_layer
        layers: List[nn.Module] = []

        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvNormActivation(
                in_channels,
                firstconv_output_channels,
                kernel_size=in_conv_kernel,
                stride=in_conv_stride,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        f_dim, t_dim = input_dims
        f_dim = cnn_out_size(f_dim, 1, 1, 3, 2)
        t_dim = cnn_out_size(t_dim, 1, 1, 3, 2)
        for cnf in inverted_residual_setting:
            f_dim = cnf.out_size(f_dim)
            t_dim = cnf.out_size(t_dim)
            cnf.f_dim, cnf.t_dim = f_dim, t_dim
            layers.append(block(cnf, se_conf, norm_layer, depthwise_norm_layer))

        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        self.embed_dim = lastconv_output_channels
        layers.append(
            ConvNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )
        self.features = nn.Sequential(*layers)

        self.head_type = head_type
        if head_type == "fully_convolutional":
            self.classifier = nn.Sequential(
                nn.Conv2d(lastconv_output_channels, num_classes, 1, 1, 0, bias=False),
                nn.BatchNorm2d(num_classes),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        elif head_type == "mlp":
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(start_dim=1),
                nn.Linear(lastconv_output_channels, last_channel),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(last_channel, num_classes),
            )
        else:
            raise NotImplementedError(
                f"Head '{head_type}' unknown; use 'mlp' or 'fully_convolutional'"
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor, return_fmaps: bool = False):
        fmaps: List[Tensor] = []
        for layer in self.features:
            x = layer(x)
            if return_fmaps:
                fmaps.append(x)
        features = F.adaptive_avg_pool2d(x, (1, 1)).squeeze()
        logits = self.classifier(x).squeeze()
        if features.dim() == 1 and logits.dim() == 1:
            features = features.unsqueeze(0)
            logits = logits.unsqueeze(0)
        if return_fmaps:
            return logits, fmaps
        return logits, features

    def forward(self, x: Tensor):
        return self._forward_impl(x)


def build_backbone(cfg: MN01Config) -> MobileNetV3:
    """mn01's MobileNetV3 (channel-wise SE, MLP classifier) at `cfg.width_mult`."""
    setting, last_channel = _mobilenet_v3_conf(width_mult=cfg.width_mult)
    se_conf = {"se_dims": [1], "se_agg": "max", "se_r": 4}  # [1] == channel axis
    return MobileNetV3(
        setting,
        last_channel,
        num_classes=cfg.num_classes,
        head_type="mlp",
        input_dims=(cfg.n_mels, cfg.n_frames),
        in_channels=1,
        se_conf=se_conf,
    )


class Mn01MelFrontend(nn.Module):
    """EfficientAT `AugmentMelSTFT` at eval settings — no SpecAugment, fixed
    fmin/fmax. Waveform (B, T) @ 32 kHz -> mel (B, n_mels, n_frames).
    """

    def __init__(self, cfg: MN01Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "preemphasis", torch.as_tensor([[[-0.97, 1.0]]]), persistent=False
        )
        self.register_buffer(
            "window",
            torch.hann_window(cfg.win_length, periodic=False),
            persistent=False,
        )
        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            cfg.n_mels,
            cfg.n_fft,
            cfg.sample_rate,
            cfg.fmin,
            cfg.fmax,
            vtln_low=100.0,
            vtln_high=-500.0,
            vtln_warp_factor=1.0,
        )
        mel_basis = F.pad(torch.as_tensor(mel_basis), (0, 1), mode="constant", value=0)
        self.register_buffer("mel_basis", mel_basis, persistent=False)

    def forward(self, waveform: Tensor) -> Tensor:
        x = F.conv1d(waveform.unsqueeze(1), self.preemphasis).squeeze(1)
        spec = torch.stft(
            x,
            self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            center=True,
            normalized=False,
            window=self.window,
            return_complex=True,
        )
        power = spec.real**2 + spec.imag**2
        melspec = torch.matmul(self.mel_basis, power)
        return (torch.log(melspec + 0.00001) + 4.5) / 5.0


class MN01Embedder(nn.Module):
    """Waveform (B, T) @ `cfg.sample_rate` -> pooled embedding (B, 96).

    The MLP classifier is kept so the released checkpoint loads with
    `strict=True`; callers use the returned `features`, which is what the
    shared ArcFace head consumes.
    """

    def __init__(self, cfg: Optional[MN01Config] = None) -> None:
        super().__init__()
        self.cfg = cfg or MN01Config()
        self.frontend = Mn01MelFrontend(self.cfg)
        self.net = build_backbone(self.cfg)

    @property
    def embed_dim(self) -> int:
        return self.net.embed_dim

    def forward(self, waveform: Tensor) -> Tensor:
        melspec = self.frontend(waveform)
        _, features = self.net(melspec.unsqueeze(1))
        return features


def load_mn01(
    cfg: Optional[MN01Config] = None,
    checkpoint: Optional[str] = None,
    map_location: str = "cpu",
) -> MN01Embedder:
    """Build mn01 and load the AudioSet weights.

    `checkpoint` (a local path) wins over `cfg.pretrained_url`; the URL path
    caches under the torch hub dir so repeated calls do not re-download.
    """
    cfg = cfg or MN01Config()
    model = MN01Embedder(cfg)
    if checkpoint is None:
        state = torch.hub.load_state_dict_from_url(
            cfg.pretrained_url, map_location=map_location, progress=False
        )
    else:
        state = torch.load(checkpoint, map_location=map_location)
    model.net.load_state_dict(state)
    return model.eval()
