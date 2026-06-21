#!/bin/bash
# gpu-pause: freeze/restore GPU process via cuda-checkpoint + (optional) CRIU disk dump.
# 详细 SOP / 限制见 SKILL.md.
set -u

DEFAULT_DUMP_ROOT=${GPU_PAUSE_DUMP_ROOT:-/tmp/gpu-pause-dumps}
SUDO=${SUDO:-sudo}
NVSMI='nvidia-smi --query-gpu=memory.used --format=csv,noheader'

usage() {
  cat <<'EOF'
gpu-pause: freeze/restore NVIDIA GPU process via cuda-checkpoint + (opt) CRIU

  status PID        check state (running / checkpointed)
  freeze PID        RAM mode: GPU released, RAM keeps process. Pair with: restore PID
  restore PID       RAM mode: re-attach GPU
  dump   PID [DIR]  disk mode: cuda-checkpoint + criu, PID killed, GPU+RAM both freed
                    DIR defaults to $GPU_PAUSE_DUMP_ROOT/pid-<PID>-<timestamp>
  undump DIR        disk mode: criu restore, PID reused (auto re-attach GPU)
  list              list dumps in $GPU_PAUSE_DUMP_ROOT
  help              this message

Required:
  cuda-checkpoint   /usr/bin/cuda-checkpoint (NVIDIA, R535+ driver)
  criu              /usr/sbin/criu (only for dump/undump; sudo apt-get install -y criu via ppa)

Env:
  GPU_PAUSE_DUMP_ROOT  default /tmp/gpu-pause-dumps
  SUDO                 override sudo prefix
EOF
}

check_cc() {
  command -v cuda-checkpoint >/dev/null || {
    echo "[FATAL] cuda-checkpoint not found; install via https://github.com/NVIDIA/cuda-checkpoint" >&2
    exit 2
  }
}
check_criu() {
  command -v criu >/dev/null || {
    echo "[FATAL] criu not found; install: sudo add-apt-repository -y ppa:criu/ppa && sudo apt-get install -y criu" >&2
    exit 2
  }
}
check_pid_alive() {
  local pid=$1
  kill -0 "$pid" 2>/dev/null || {
    echo "[FATAL] PID $pid not alive" >&2
    exit 2
  }
}

cmd=${1:-help}; shift 2>/dev/null || true

case "$cmd" in
  status)
    check_cc
    PID=${1:-}; [ -z "$PID" ] && { echo "usage: gpu-pause status PID"; exit 1; }
    check_pid_alive "$PID"
    echo "PID $PID state: $(cuda-checkpoint --get-state --pid "$PID")"
    eval "$NVSMI"
    ;;

  freeze)
    check_cc
    PID=${1:-}; [ -z "$PID" ] && { echo "usage: gpu-pause freeze PID"; exit 1; }
    check_pid_alive "$PID"
    state=$(cuda-checkpoint --get-state --pid "$PID")
    [ "$state" = "running" ] || { echo "[FAIL] PID $PID state=$state (need running)"; exit 1; }
    before=$(eval "$NVSMI")
    cuda-checkpoint --toggle --pid "$PID" || { echo "[FAIL] toggle failed"; exit 3; }
    sleep 1
    after=$(eval "$NVSMI")
    echo "[OK] PID $PID  $state -> $(cuda-checkpoint --get-state --pid "$PID")"
    echo "     GPU mem: $before -> $after"
    echo "     restore: bash $0 restore $PID"
    ;;

  restore)
    check_cc
    PID=${1:-}; [ -z "$PID" ] && { echo "usage: gpu-pause restore PID"; exit 1; }
    check_pid_alive "$PID"
    state=$(cuda-checkpoint --get-state --pid "$PID")
    [ "$state" = "checkpointed" ] || { echo "[FAIL] PID $PID state=$state (need checkpointed)"; exit 1; }
    before=$(eval "$NVSMI")
    cuda-checkpoint --toggle --pid "$PID" || { echo "[FAIL] toggle failed"; exit 3; }
    sleep 1
    after=$(eval "$NVSMI")
    echo "[OK] PID $PID  $state -> $(cuda-checkpoint --get-state --pid "$PID")"
    echo "     GPU mem: $before -> $after"
    ;;

  dump)
    check_cc; check_criu
    PID=${1:-}; [ -z "$PID" ] && { echo "usage: gpu-pause dump PID [DIR]"; exit 1; }
    check_pid_alive "$PID"
    DIR=${2:-$DEFAULT_DUMP_ROOT/pid-$PID-$(date +%Y%m%d-%H%M%S)}
    mkdir -p "$DIR"
    echo "[dump] PID=$PID  ->  $DIR"
    cuda-checkpoint --action=lock       --pid "$PID" || { echo "[FAIL] lock";       exit 3; }
    cuda-checkpoint --action=checkpoint --pid "$PID" || { echo "[FAIL] checkpoint"; exit 3; }
    sleep 1
    [ "$(cuda-checkpoint --get-state --pid "$PID")" = "checkpointed" ] || { echo "[FAIL] state not checkpointed"; exit 3; }
    eval "$NVSMI"
    # criu dump (PID killed by default)
    $SUDO criu dump --tree "$PID" --images-dir "$DIR" --shell-job --log-file "$DIR/dump.log" || {
      echo "[FAIL] criu dump failed; see $DIR/dump.log"; exit 4;
    }
    # 验证 PID 真死
    if kill -0 "$PID" 2>/dev/null; then
      echo "[WARN] PID $PID still alive after criu dump"
    else
      echo "[OK] PID killed"
    fi
    size=$(du -sh "$DIR" | cut -f1)
    n_files=$(find "$DIR" -type f | wc -l)
    echo "[done] dump $size in $n_files files at $DIR"
    eval "$NVSMI"
    echo "[restore later] bash $0 undump $DIR"
    ;;

  undump)
    check_criu
    DIR=${1:-}; [ -z "$DIR" ] && { echo "usage: gpu-pause undump DUMP_DIR"; exit 1; }
    [ -d "$DIR" ] || { echo "[FAIL] no such dir: $DIR"; exit 1; }
    echo "[restore] $DIR"
    $SUDO criu restore --images-dir "$DIR" --shell-job --log-file "$DIR/restore.log" -d || {
      echo "[FAIL] criu restore failed; see $DIR/restore.log"; exit 4;
    }
    sleep 2
    # CRIU 默认 reuse 原 PID, 从 dump 文件提取
    PID=$(ls "$DIR" | grep -oP 'core-\K[0-9]+(?=\.img)' | sort -n | head -1)
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
      echo "[OK] PID $PID restored"
      ps -p "$PID" -o pid,stat,etime,cmd | head -2
      eval "$NVSMI"
      # CRIU restore 已自动激活 GPU; cuda-checkpoint --get-state 在某些场景报错但进程实际 running, ignore non-zero exit
      cuda-checkpoint --get-state --pid "$PID" 2>/dev/null || true
    else
      echo "[WARN] cannot detect restored PID from $DIR"
      ps aux | grep -E '(criu|python)' | grep -v grep | tail -5
    fi
    ;;

  list)
    [ -d "$DEFAULT_DUMP_ROOT" ] || { echo "no dumps in $DEFAULT_DUMP_ROOT"; exit 0; }
    for d in "$DEFAULT_DUMP_ROOT"/*/; do
      [ -d "$d" ] || continue
      size=$(du -sh "$d" 2>/dev/null | cut -f1)
      mtime=$(stat -c '%y' "$d" | cut -d. -f1)
      echo "$size  $mtime  $d"
    done
    ;;

  help|--help|-h|*)
    usage
    ;;
esac
