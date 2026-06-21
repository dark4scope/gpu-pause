#!/bin/bash
# gpu-pause 一键安装脚本.
# 1. 装 cuda-checkpoint binary 到 /usr/bin (sudo)
# 2. 装本仓库 gpu-pause.sh 到 /usr/local/bin (sudo)
# 3. 装 Claude Code skill 到 ~/.claude/skills/gpu-pause (可选)
# 4. (可选) 装 CRIU
set -euo pipefail

REPO_URL=${REPO_URL:-https://github.com/dark4scope/gpu-pause}
RAW_URL=${RAW_URL:-https://raw.githubusercontent.com/dark4scope/gpu-pause/main}
SKILL_DIR=${SKILL_DIR:-$HOME/.claude/skills/gpu-pause}
INSTALL_PREFIX=${INSTALL_PREFIX:-/usr/local/bin}

echo "=========================================="
echo " gpu-pause installer"
echo "=========================================="
echo

# Step 1: cuda-checkpoint
echo "[1/4] cuda-checkpoint (NVIDIA, required)"
if command -v cuda-checkpoint >/dev/null; then
  echo "    already installed: $(cuda-checkpoint --help | head -2 | tail -1)"
else
  echo "    NVIDIA driver:"
  nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | sed 's/^/      /'
  echo "    downloading binary..."
  curl -fsSL https://github.com/NVIDIA/cuda-checkpoint/raw/main/bin/x86_64_Linux/cuda-checkpoint -o /tmp/cuda-checkpoint
  sudo install -m 755 /tmp/cuda-checkpoint /usr/bin/cuda-checkpoint
  rm /tmp/cuda-checkpoint
  echo "    installed: $(cuda-checkpoint --help | head -2 | tail -1)"
fi
echo

# Step 2: gpu-pause.sh
echo "[2/4] gpu-pause.sh → $INSTALL_PREFIX/gpu-pause"
if [ -f scripts/gpu-pause.sh ]; then
  SRC=scripts/gpu-pause.sh
else
  echo "    fetching from $RAW_URL ..."
  curl -fsSL $RAW_URL/scripts/gpu-pause.sh -o /tmp/gpu-pause.sh
  SRC=/tmp/gpu-pause.sh
fi
sudo install -m 755 $SRC $INSTALL_PREFIX/gpu-pause
gpu-pause help > /dev/null && echo "    OK ($(which gpu-pause))"
echo

# Step 3: Claude Code skill
echo "[3/4] Claude Code skill (optional)"
if [ -d "$HOME/.claude" ]; then
  mkdir -p $SKILL_DIR/scripts
  if [ -f SKILL.md ]; then
    cp SKILL.md $SKILL_DIR/
    cp scripts/gpu-pause.sh $SKILL_DIR/scripts/
  else
    curl -fsSL $RAW_URL/SKILL.md -o $SKILL_DIR/SKILL.md
    curl -fsSL $RAW_URL/scripts/gpu-pause.sh -o $SKILL_DIR/scripts/gpu-pause.sh
  fi
  chmod +x $SKILL_DIR/scripts/gpu-pause.sh
  echo "    installed to $SKILL_DIR"
else
  echo "    ~/.claude not found, skip"
fi
echo

# Step 4: CRIU (optional)
echo "[4/4] CRIU (only needed for DISK mode dump/undump)"
if command -v criu >/dev/null; then
  echo "    already installed: $(criu --version | head -1)"
else
  echo "    not installed."
  if [ -f /etc/lsb-release ] && grep -q Ubuntu /etc/lsb-release; then
    read -p "    install CRIU via ppa:criu/ppa ? (y/N): " yn
    if [ "${yn:-N}" = "y" ] || [ "${yn:-N}" = "Y" ]; then
      sudo add-apt-repository -y ppa:criu/ppa
      sudo apt-get update
      sudo apt-get install -y criu
      sudo criu check && echo "    criu OK"
    else
      echo "    skipped — DISK mode (dump/undump) will not work"
    fi
  else
    echo "    please install criu manually for your distro"
  fi
fi
echo

echo "=========================================="
echo " DONE"
echo "=========================================="
echo "  shell:        gpu-pause status <PID>"
echo "  Claude Code:  $SKILL_DIR (auto-triggered)"
echo "  docs:         $REPO_URL"
