#!/usr/bin/env bash
# One-time setup for driving a real Colab GPU runtime from this machine via the
# `colab` CLI (https://github.com/googlecolab/google-colab-cli).
#
# Ported from the sibling bicafe project.
#
# Why not just `uv tool install google-colab-cli` and go: as of
# google-colab-cli==0.6.0, its `jupyter-kernel-client` dependency is unpinned,
# and the latest release (1.0.1) renamed `KernelClient` ->
# `JupyterKernelClient`, breaking every `colab exec`/`repl`/`run` call with
# `AttributeError: module 'jupyter_kernel_client' has no attribute
# 'KernelClient'`. Pin to 0.15.0 (last pre-rename release) until upstream fixes
# the pin.
#
# Why ADC over the default oauth2 flow: oauth2's InstalledAppFlow mints a
# PKCE code_challenge per process invocation, so a code obtained against one
# `colab new` call cannot be redeemed by a later, separate invocation -- fatal
# for non-interactive/agent-driven use. ADC's login is a single one-shot
# `gcloud` command independent of which `colab` invocation reads the credential
# file afterwards.
set -euo pipefail

GCLOUD_DIR="$HOME/google-cloud-sdk"

if ! command -v gcloud >/dev/null 2>&1 && [ ! -x "$GCLOUD_DIR/bin/gcloud" ]; then
  echo "[colab_setup] Installing Google Cloud SDK to $GCLOUD_DIR (no sudo)..."
  curl -s https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir="$HOME"
fi
export PATH="$GCLOUD_DIR/bin:$PATH"

if ! command -v colab >/dev/null 2>&1; then
  echo "[colab_setup] Installing google-colab-cli, pinned to a working jupyter-kernel-client..."
  uv tool install google-colab-cli --with "jupyter-kernel-client==0.15.0" --force
fi

ADC_FILE="$HOME/.config/gcloud/application_default_credentials.json"
if [ ! -f "$ADC_FILE" ]; then
  echo "[colab_setup] No ADC credentials found -- running the one-time login."
  echo "[colab_setup] This needs the 4 scopes the Colab backends require;"
  echo "[colab_setup] a partial scope set 403s later with a confusing error."
  gcloud auth application-default login \
    --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
else
  echo "[colab_setup] ADC credentials already present at $ADC_FILE"
fi

echo "[colab_setup] Verifying..."
colab --auth=adc whoami
echo "[colab_setup] Done. Use --auth=adc on every colab_* script (already wired in)."
