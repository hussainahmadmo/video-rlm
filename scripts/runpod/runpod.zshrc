# VIMIO / video-rlm RunPod zsh setup.
# This file is safe to commit. Put secrets in ~/.vimio_secrets, not here.

autoload -Uz colors && colors

typeset -U path PATH

parse_git_branch() {
  git branch 2>/dev/null | sed -n '/\* /s///p' | awk '{print "("$1")"}'
}

setopt prompt_subst
PROMPT='%F{green}%n@%m %F{blue}%~%f %F{red}$(parse_git_branch)%f %# '

HISTSIZE=1000
SAVEHIST=2000
HISTFILE=~/.zsh_history
setopt APPEND_HISTORY
setopt HIST_IGNORE_DUPS
setopt HIST_IGNORE_SPACE

alias ls='ls --color=auto'
alias grep='grep --color=auto'
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

export VIMIO_ROOT="${VIMIO_ROOT:-/workspace/video-rlm}"
export VIMIO_CACHE_ROOT="${VIMIO_CACHE_ROOT:-/workspace/.cache/vimio}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_METRICS_CACHE="${HF_METRICS_CACHE:-$HF_HOME/metrics}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_METRICS_CACHE" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$VIMIO_CACHE_ROOT" 2>/dev/null || true

# Optional local secrets. This file should not be committed.
# Example:
#   export HF_TOKEN=...
#   export WANDB_API_KEY=...
if [ -f "$HOME/.vimio_secrets" ]; then
  source "$HOME/.vimio_secrets"
fi

if command -v pyenv >/dev/null 2>&1; then
  eval "$(pyenv init -)"
fi

for conda_sh in \
  "$HOME/miniconda3/etc/profile.d/conda.sh" \
  "/workspace/miniconda3/etc/profile.d/conda.sh" \
  "/opt/conda/etc/profile.d/conda.sh" \
  "/dataheart/hussainahmad/miniconda3/etc/profile.d/conda.sh"
do
  if [ -f "$conda_sh" ]; then
    source "$conda_sh"
    export CONDA_SOURCED_FROM="${conda_sh%/etc/profile.d/conda.sh}"
    break
  fi
done

if [ -d "/usr/local/cuda" ]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
elif [ -d "/usr/local/cuda-12.4" ]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
fi

if [ -n "${CUDA_HOME:-}" ]; then
  export CUDAToolkit_ROOT="$CUDA_HOME"
  path=("$CUDA_HOME/bin" $path)
  typeset -U ld_library_path
  ld_library_path=()
  if [[ -n "$LD_LIBRARY_PATH" ]]; then
    ld_library_path=(${(s.:.)LD_LIBRARY_PATH})
  fi
  ld_library_path=("$CUDA_HOME/lib64" ${ld_library_path:#})
  export LD_LIBRARY_PATH="${(j.:.)ld_library_path}"
fi

if [ -f "$HOME/.zsh/zsh-autocomplete/zsh-autocomplete.plugin.zsh" ]; then
  source "$HOME/.zsh/zsh-autocomplete/zsh-autocomplete.plugin.zsh"
fi

if [ -d "$VIMIO_ROOT" ]; then
  cd "$VIMIO_ROOT"
fi

hash -r
