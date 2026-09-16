# Compute / Hardware Plan

What to rent for the experiment matrix, why, and how fast it should be. Written
for the "which GPU do we pay for" decision; the runbook itself is
[`EXPERIMENTS.md`](EXPERIMENTS.md).

## Workload profile

- STgram-MFN is **1.16 M params but compute-bound on the input**, not on
  parameters: 10 s @ 16 kHz → 2 × 128 × 313 activations through MobileFaceNet.
  Measured peak is ~96 MB/sample, so batch 128 ≈ **12 GB VRAM**.
- The local analysis found **batch size is not the lever** (60 → 62 clips/s from
  batch 16 → 32); raw FLOPs and data feeding are.
- Data: DCASE 2020 dev + eval ≈ **14 GB**.
- Training set: ~36 k clips/epoch (dev + eval-normal).

## Speed levers (ranked)

| # | Lever | Speedup | Notes |
|---|---|---|---|
| 1 | Modern GPU (Ada/Ampere) + AMP | ~20–50× vs P2000 | Pascal/Volta have no tensor cores → fp32-only |
| 2 | Fewer epochs (300 → 60) | 5× | trajectory check first; `configs/train_t4_long.yaml` |
| 3 | Data pipeline (workers + NVMe) | unlocks #1 | at high clip rates the WAV decode + mel becomes the bottleneck |
| 4 | `torch.compile` + channels-last + `cudnn.benchmark` | ~1.3–2× | not wired into `train.py` yet |
| 5 | 2–4× GPU DDP | ~2–4× | marginal for a 1.16 M model; costs more |

## Recommended config

| Resource | Minimum | Ideal | Why |
|---|---|---|---|
| GPU | RTX A4000 16 GB (Ampere) | **RTX 4090 / A5000 / 3090 24 GB** | batch 128 fits 16 GB, 24 GB has headroom |
| vCPU | 4 | **8–16** | 16 kHz audio decode in DataLoader workers |
| RAM | 16 GB | **32–48 GB** | workers + dataset (48 GB allows `load_in_memory`) |
| Disk | 100 GB | **300 GB NVMe** | ~14 GB data + venv + HF cache + checkpoints |
| OS | Ubuntu 22.04/24.04 | **Ubuntu 22.04** | driver/CUDA simplicity |
| Network | 100 Mbps | **1 Gbps** | ~14 GB HF pull |
| Access | SSH + sudo | **root SSH + 24/7** | needed to drive the runs |

**Avoid:** P40 / V100 (no AMP), 12 GB consumer cards, browser-only "cloud
gaming" tiers, and spot/interruptible instances for the long runs.

**Driver/CUDA:** the repo pins `torch` **cu126** (the newest tag with `sm_61`
for the local P2000). Ada (`sm_89`, RTX 4090) and Ampere (`sm_86`, 3090/A5000)
run clean on it. **Avoid RTX 5090** (Blackwell `sm_120`) — it needs CUDA 12.8+ /
newer PyTorch and will fight the pin.

## Provider shortlist

**VN-facing (VND, local support):**
- **Ckey.vn** — marketplace (1,500+ GPUs); legit (registered household business,
  MoIT-confirmed), VN bank + WebMoney, Telegram/Zalo/Discord. Pick a
  **Datacenter** host, not Community.
- VNG Cloud — RTX 4090, A40 48 GB, L40S, H100; pay-as-you-go.
- Sunteco — T4 ~15k, 4090 ~40k, A100 ~50k, H100 ~80k VND/h (pricier).
- ThueGPU — P40 8k, A4000 10k, 3090 16k, V100 25k VND/h.

**International (card payment, reliable):**
- Hyperstack — **A4000 16 GB $0.15/h** (per-minute, SOC2).
- RunPod **Secure** — A5000 24 GB $0.27/h, 3090 $0.50, 4090 $0.74.
- TensorDock — RTX 4090 $0.35/h, consumer from $0.12; KVM root, 99.99% uptime.
- Vast.ai on-demand / JarvisLabs L4/A30.

## Ckey.vn cart

- Product: **Thuê Máy → KVM GPU** (full VM + root, not Container, not the web PC).
- **1× RTX 4090 24 GB** (fallback: RTX 3090 24 GB).
- **≥8 vCPU · ≥32 GB RAM · ≥200 GB NVMe · ≥1 Gbps · Ubuntu 22.04 · root SSH**.
- Host type **Datacenter**, high uptime.

Confirm with the host before paying: dedicated/passthrough GPU, root SSH, NVIDIA
driver ≥ 550 installable, and the exact vCPU/RAM/disk on the listing. Storage is
wiped when the rental ends → checkpoints stream to HF continuously.

## Budget & expected wall-clock

| Runtime | Baseline (300 ep) | Trajectory (60 ep) |
|---|---|---|
| Local P2000 (fp32) | ~50 h | ~10 h |
| RTX 4090 + AMP | ~1–2 h | ~20–30 min |
| RTX 3090 + AMP | ~2–3 h | ~30–45 min |

At 4090 market rates the full 2×3 matrix is roughly **$5–15 all-in**.

## Fidelity caveat

AMP + fewer epochs + cached features all move the run away from the published
fp32 / 300-epoch protocol. Run the **M1 gate** faithful (fp32) if the goal is to
reproduce 92.36 AUC; for the **A-vs-B comparison** the fast protocol is fine —
just use the *same* one for both backbones and log it.
