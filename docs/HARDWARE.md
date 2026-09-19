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

## RunPod — chosen rental (2026-09-19)

Ckey.vn is down (backend DB unreachable since 2026-09-19 02:22 UTC — every route
returns `Không thể kết nối đến cơ sở dữ liệu`), so the run moves to RunPod. Live
availability/prices via RunPod's GraphQL `gpuTypes.lowestPrice`.

**Pick: 1× RTX A5000 24 GB** (Ampere `sm_86`, runs the `cu126` pin; ~$0.27/h
Secure). Fallbacks if A5000 has no capacity (it periodically drops off the
listing): **RTX A4500 20 GB @ ~$0.19/h**, **RTX 3090 24 GB @ ~$0.22/h**, or
**RTX 4000 Ada 20 GB @ ~$0.20/h**. Avoid all RTX 5090/5080/5070 (Blackwell
`sm_120` fights the pin).

### Pod configuration

| Field | Value |
|---|---|
| GPU | RTX A5000 24 GB (fallback A4500 / 3090) |
| Cloud type | Secure Cloud |
| Template | RunPod PyTorch 2.8.0 — **CUDA 12.8** (not 13.0; matches the `cu126` major) |
| Container disk | 40 GB — image ~12 GB + data 17 GB + venv 6.8 GB. Use `uv sync --no-cache` (the uv wheel cache alone is 7 GB and would overflow) |
| Expose HTTP ports | 8888 (Jupyter), 6006 (TensorBoard) |
| Expose TCP port | 22 (enables full SSH + scp/sftp) |
| Env vars | `WANDB_API_KEY`, `HF_TOKEN` (`hub_utils.py:15` accepts either) |
| SSH key | add `~/.ssh/id_ed25519.pub` to RunPod account → SSH Public Keys |

### Setup (SSH over exposed TCP)

```bash
ssh root@<pod-ip> -p <ssh-port> -i ~/.ssh/id_ed25519

curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
git clone https://github.com/LakoreAI/Research_AnomalySoundDetection.git
cd Research_AnomalySoundDetection
uv sync --no-cache --extra wandb --extra hub && uv run pytest

# private HF data mirrors (faster than Zenodo; see hf-dataset-fast-resume skill)
uv run python -c "
from huggingface_hub import snapshot_download
for local, repo in [('data/raw','LakoreAI/stgram-mfn-dcase2020-dev'),
                    ('data/raw_eval','LakoreAI/stgram-mfn-dcase2020-eval')]:
    snapshot_download(repo_id=repo, repo_type='dataset', local_dir=local, max_workers=16)
"
uv run python scripts/data/prepare_data.py --root data/raw --check
```

### Training run

Same recipe as the Ckey block below (24 GB → no grad accumulation). Run inside
`tmux`; `--hf_push_repo` survives pod termination.

```bash
uv run python scripts/training/train.py --config configs/train_t4_long.yaml \
  --epochs 60 --add_root data/raw_eval \
  --batch_size 128 --accum_steps 1 --num_workers 8 --pin_memory --amp \
  --hf_push_repo LakoreAI/stgram-mfn-runpod-a5000
```

Cost at $0.27/h: 60-epoch ≈ $0.15–0.20, 300-epoch ≈ $0.60, full matrix ≈ $2–8.

## Ckey.vn — chosen rental (2026-09-18) · SUPERSEDED

Live listing checked against the workload profile above; priced in VND/h, ordered
by value. **Pick: Ckey GPU3 listing `106600`** (start ≈ 6,563 VND/h ≈ $0.25/h —
cheapest machine that clears every minimum; 24 GB holds reference batch 128 in
one pass):

| Spec | Value |
|---|---|
| GPU | **1× RTX 3090 24 GB** (Ampere `sm_86`; runs the `cu126` pin) |
| CPU / RAM | Intel i5-14400F, 8c/16t / 64 GB |
| Disk / Net | 378 GB NVMe (2,894 MB/s) / 452↓·386↑ Mbps |
| Driver | CUDA max 13.0 (≥ 12.x needed for `cu126`) |
| Uptime / max rent | 99.99% / 1,440 h |
| Price | 6,563 VND/h · 157,505 VND/day |

Fallback / speed pick: listing `50256` — 1× RTX 4090 24 GB (Ryzen 9 7950X
16c/32t, 32 GB, 353 GB, 15,313 VND/h) for the faithful 300-epoch gate.
**Avoid:** all RTX 5090/5080/5070 (Blackwell `sm_120` fights the `cu126` pin),
P40/V100/P2000 (no tensor cores → no AMP), <16 GB cards, and explicitly-Community
hosts when a bare-metal box exists.

### Order settings (page `/thanh-toan-gpu3/106600`)

- Duration: **24 h** to start (renew before expiry — data is wiped on
  expiry/rebuild).
- OS: **Ubuntu 22.04** (24.04 fallback) with the NVIDIA driver preinstalled.
- Access: **root SSH**.
- Port-forward (optional): **8888** Jupyter, **6006** TensorBoard. W&B needs no
  inbound.

### Machine setup (root SSH)

```bash
apt-get update && apt-get install -y git curl
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
nvidia-smi                              # confirm RTX 3090 + driver

git clone <repo-url> && cd Research_AnomalySoundDetection
uv sync --extra wandb --extra hub       # torch cu126 wheels
uv run pytest

# DCASE 2020 dev + eval (or restore from the HF dataset copy)
uv run python scripts/data/download_data.py --dataset dcase2020      --dest data/raw
uv run python scripts/data/download_data.py --dataset dcase2020-eval --dest data/raw_eval
uv run python scripts/data/prepare_data.py --root data/raw --check
```

### Training run (24 GB → no grad accumulation)

`configs/train.yaml` is sized for the 4 GB P2000 (batch 16 × accum 8). On the
3090, override: batch 128 ≈ 12 GB VRAM, effective batch 128 = reference.

```bash
# 60-epoch trajectory (fast path)
uv run python scripts/training/train.py --config configs/train_t4_long.yaml \
  --epochs 60 --add_root data/raw_eval \
  --batch_size 128 --accum_steps 1 --num_workers 8 --pin_memory --amp \
  --hf_push_repo LakoreAI/stgram-mfn-rtx3090

# faithful 300-epoch gate
uv run python scripts/training/train.py --config configs/train.yaml \
  --add_root data/raw_eval \
  --batch_size 128 --accum_steps 1 --num_workers 8 --pin_memory --amp \
  --hf_push_repo LakoreAI/stgram-mfn-rtx3090
```

`--amp` activates tensor cores (`src/pipelines/config.py:40`,
`src/pipelines/train.py:304`). `--hf_push_repo` streams `best.pt` + `epoch_N.pt`
to HF every N epochs and auto-resumes on startup
(`src/pipelines/train.py:350`) — mandatory because Ckey wipes disk on
expiry/rebuild.

## Budget & expected wall-clock

| Runtime | Baseline (300 ep) | Trajectory (60 ep) |
|---|---|---|
| Local P2000 (fp32) | ~50 h | ~10 h |
| RTX 4090 + AMP | ~1–2 h | ~20–30 min |
| RTX 3090 + AMP | ~2–3 h | ~30–45 min |

At 4090 market rates the full 2×3 matrix is roughly **$5–15 all-in**. At the
chosen listing `106600` (~$0.25/h) the same matrix is ≈ **$3–5**; tomorrow's
first 60-epoch run is ≈ **$1**.

## Fidelity caveat

AMP + fewer epochs + cached features all move the run away from the published
fp32 / 300-epoch protocol. Run the **M1 gate** faithful (fp32) if the goal is to
reproduce 92.36 AUC; for the **A-vs-B comparison** the fast protocol is fine —
just use the *same* one for both backbones and log it.
