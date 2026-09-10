# RunPod Setup

Use this after cloning the repository on a fresh RunPod instance.

```bash
cd /workspace
git clone git@github.com:hussainahmadmo/video-rlm.git
cd video-rlm
VLLM_VERSION=0.17.0 bash scripts/runpod/install_runpod.sh
```

If SSH is not configured on the pod:

```bash
git clone https://github.com/hussainahmadmo/video-rlm.git
```

The installer creates or reuses a conda env named `vllm-mm`, installs FFmpeg/build tools when running as root, installs Python dependencies, installs the local `decord` package when possible, and installs `vllm` by default.

Useful toggles:

```bash
INSTALL_VLLM=0 bash scripts/runpod/install_runpod.sh
INSTALL_YOLO=1 bash scripts/runpod/install_runpod.sh
INSTALL_ZSH=0 bash scripts/runpod/install_runpod.sh
INSTALL_ZSH_AUTOCOMPLETE=0 bash scripts/runpod/install_runpod.sh
ENV_NAME=vimio PYTHON_VERSION=3.10 bash scripts/runpod/install_runpod.sh
```

The installer also copies `scripts/runpod/runpod.zshrc` to `~/.zshrc` and backs up an existing file as `~/.zshrc.before-vimio`. It keeps the useful local shell setup but intentionally does not include secrets. Put tokens in `~/.vimio_secrets`:

```bash
cat > ~/.vimio_secrets <<'EOF'
export HF_TOKEN=...
export WANDB_API_KEY=...
EOF
chmod 600 ~/.vimio_secrets
```

Then start zsh:

```bash
zsh
```

After install:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm-mm
cd /workspace/video-rlm
```

## A40 GPU-saturated fairness experiment

Copy one ordinary MP4 to the pod. The experiment deliberately repeats that
same video for both tenants so host preparation is matched and the only
controlled difference is 512 versus 64 forced output tokens. The launcher
profiles token-service cost on the A40 before running the matched policies;
do not copy a profile measured on another GPU.

```bash
chmod +x run_gpu_saturated_fairness_with_server.sh \
  scripts/runpod/run_gpu_saturated_fairness_a40.sh

VIDEO_FILE=/workspace/data/gpu-fairness-source.mp4 \
  scripts/runpod/run_gpu_saturated_fairness_a40.sh
```

The default run uses Qwen2.5-VL-7B-Instruct on one A40, five policies, three
matched seeds, four concurrent vLLM requests, eight preparation workers, and
a 30-second overloaded arrival window. A short smoke test is:

```bash
VIDEO_FILE=/workspace/data/gpu-fairness-source.mp4 \
SEEDS=1 POLICIES='fcfs max_min' DURATION_S=10 \
  scripts/runpod/run_gpu_saturated_fairness_a40.sh
```

Results are written below
`large_sweeps/gpu_saturated_fairness_a40_qwen7b/`; server and GPU-monitor logs
are written below `logs/gpu_saturated_fairness_a40_qwen7b/`.

Check that the pushed caption caches are present:

```bash
find conductor/experiments/self_improving/data/video_agent_caption_cache* -type f | wc -l
du -sh conductor/experiments/self_improving/data/video_agent_caption_cache*
```

Example vLLM servers:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 9000 \
  --model Qwen/Qwen2.5-VL-7B-Instruct

CUDA_VISIBLE_DEVICES=1 \
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 9001 \
  --model Qwen/Qwen2.5-VL-7B-Instruct
```

Example cached agentic run:

```bash
python -m conductor.control_plane.run_video_agent \
  --force-agentic \
  --dedupe-input \
  --input conductor/experiments/self_improving/data/questions_large_train_val_test_unique_no_vrbench.jsonl \
  --sweep-results conductor/experiments/self_improving/data/fixed_results_large_train_val_8configs_with_dataset.jsonl \
  --output conductor/experiments/self_improving/data/eval_agentic_runpod.jsonl \
  --base-url http://localhost:9000/v1 \
  --caption-base-url http://localhost:9001/v1 \
  --vlm-model Qwen/Qwen2.5-VL-7B-Instruct \
  --caption-model Qwen/Qwen2.5-VL-7B-Instruct \
  --caption-prompt-style videoagent2 \
  --context-coverage all \
  --context-frames-per-segment 2 \
  --caption-workers 8 \
  --answer-confidence-threshold 5 \
  --max-rounds 5
```
