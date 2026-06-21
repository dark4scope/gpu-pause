# gpu-pause

> **Freeze a running NVIDIA GPU training process, release the GPU, restore later with byte-exact state.**

Wraps NVIDIA's [`cuda-checkpoint`](https://github.com/NVIDIA/cuda-checkpoint) (shipped 2024.4) + optional [CRIU](https://criu.org) for full disk dump. Verified byte-exact: model parameters hash + optimizer state hash + forward output stay identical across freeze/restore.

---

## Use cases

- Mid-training, need GPU for a quick eval / inference / share with teammate → **freeze** then `restore`
- Want to shut down the machine overnight → **dump** to disk, `undump` next morning
- Single GPU rotating between multiple jobs

## How it works

```
┌─────────────────────────────────────────────────────────────────┐
│                       training process (PID)                    │
│  Python ─→ PyTorch ─→ CUDA runtime ─→ Driver ─→ 36 GB GPU mem   │
└─────────────────────────────────────────────────────────────────┘

  ↓ cuda-checkpoint --action=checkpoint   (RAM mode)

┌─────────────────────────────────────────────────────────────────┐
│                       training process (PID)                    │
│  Python ─→ PyTorch ─→ [GPU state swapped to CPU RAM, GPU freed] │
└─────────────────────────────────────────────────────────────────┘
                          ↓
                  ⚡ GPU mem = 0 MiB, free for others ⚡

  ↓ criu dump                            (DISK mode further)

┌─────────────────────────────────────────────────────────────────┐
│        on-disk dump 1.2 GB ~ 50 GB                              │
│   /tmp/gpu-pause-dumps/pid-12345-20260621-2230/                 │
│   ├── pages-*.img   (process RAM + GPU contents)                │
│   ├── core-*.img    (CPU regs, thread state)                    │
│   └── ... (43 files)                                            │
└─────────────────────────────────────────────────────────────────┘
                          ↓
              ⚡ GPU + RAM both free, machine can reboot ⚡

  ↓ criu restore (cuda-checkpoint restore auto-triggered)

restored: same PID, GPU auto re-attached, byte-exact continuation
```

## Install

### 1. cuda-checkpoint (required)

```bash
# Requires NVIDIA driver R535+ (CUDA 12.4+)
nvidia-smi | head -3

curl -sL https://github.com/NVIDIA/cuda-checkpoint/raw/main/bin/x86_64_Linux/cuda-checkpoint -o /tmp/cuda-checkpoint
sudo install -m 755 /tmp/cuda-checkpoint /usr/bin/cuda-checkpoint
cuda-checkpoint --help | head -2
```

### 2. CRIU (only for DISK mode)

Ubuntu 22.04 / 24.04: must use PPA (system apt is obsoleted):

```bash
sudo add-apt-repository -y ppa:criu/ppa
sudo apt-get update && sudo apt-get install -y criu
sudo criu check  # must output "Looks good"
```

### 3. This script

```bash
git clone https://github.com/dark4scope/gpu-pause.git
cd gpu-pause
sudo install -m 755 scripts/gpu-pause.sh /usr/local/bin/gpu-pause
gpu-pause help
```

One-liner:

```bash
curl -sL https://raw.githubusercontent.com/dark4scope/gpu-pause/main/install.sh | bash
```

## Usage

```
gpu-pause <action> [args]

  status PID        check state (running / checkpointed)
  freeze PID        RAM mode: release GPU, process sleeps in RAM
  restore PID       RAM mode: re-attach GPU, process continues
  dump   PID [DIR]  DISK mode: full dump, PID killed, GPU+RAM both freed
                    DIR defaults to /tmp/gpu-pause-dumps/pid-<PID>-<timestamp>
  undump DIR        DISK mode: criu restore, original PID reused, GPU auto-attaches
  list              list all dumps
  help              this message
```

### Example 1: RAM mode (most common)

```bash
$ gpu-pause freeze 12345
[OK] PID 12345  running -> checkpointed
     GPU mem: 36251 MiB -> 1 MiB
     restore: bash gpu-pause restore 12345

# GPU is free now -- run something else ...

$ gpu-pause restore 12345
[OK] PID 12345  checkpointed -> running
     GPU mem: 1 MiB -> 36254 MiB
```

Process continues from the exact step it was frozen at — **model params, optimizer m+v, RNG state all byte-exact identical**, as if the freeze never happened.

### Example 2: DISK mode (full hibernate)

```bash
$ gpu-pause dump 12345
[done] dump 1.2G in 49 files at /tmp/gpu-pause-dumps/pid-12345-20260621-2230

# Process killed, GPU+RAM both released, machine can reboot

$ gpu-pause undump /tmp/gpu-pause-dumps/pid-12345-20260621-2230
[OK] PID 12345 restored
```

PID stays the same (CRIU reuses original PID by default).

## Verification

`examples/verify_byte_exact.py` runs a small PyTorch model + AdamW, enters a forward loop maintaining GPU activity, and automatically verifies:

- ✅ `model.parameters` hash identical
- ✅ `optimizer.state` hash identical
- ✅ Same fixed input → identical forward output (byte-exact)
- ✅ Counter tensor continues from N+1 after freeze at N

Measured timeline (RTX 3090 + driver R595):

| T | Event | GPU mem | Counter | PID state |
|---|---|---|---|---|
| T0 | normal | 512 MiB | 27 | running |
| T+1s | `gpu-pause freeze` | **1 MiB** | 27 | **checkpointed** |
| T+20s | wait, GPU free | 1 MiB | 27 | sleep in RAM |
| T+21s | `gpu-pause restore` | 512 MiB | 27 | running |
| T+22s | continue | 153 MiB | **28** | running |

## Limitations / Risks

### RAM mode

- ✅ Verified byte-exact (single-threaded PyTorch + Linear + AdamW)
- ✅ **Verified working** with `SFTTrainer` + unsloth + flash-attn + DL_WORKERS=4 + grad_ckpt (2026-06-21 on Qwen3.5-0.8B SFT; GPU 4438 MiB → 4 MiB, training resumed seamlessly step 86 → 117)
- ❌ Process RAM = original RAM + GPU mem (9B SFT needs +36 GB RAM)
- ❌ Long sleep (hours+) may trigger kernel reaper / heartbeat issues

### DISK mode

- ✅ Verified byte-exact for single-threaded Python + PyTorch
- ✅ GPU + RAM both freed, machine can reboot (within same mount/namespace)
- ❌ **Verified FAIL** on `SFTTrainer + DL_WORKERS > 0` (2026-06-21): PyTorch DataLoader **child processes** (`pt_data_worker`) hold CUDA mappings (`0x200200000` device memory range). CRIU's `cuda_plugin` cannot dump non-regular mappings → `Dumping FAILED`
- ⚠️ Workaround: use `DataLoader(num_workers=0)` (single process, no children) — costs ~30% data loading speed for CRIU compatibility
- ❌ Dump size = process RAM + GPU mem (9B ≈ 40-50 GB)
- ❌ Cross-machine restore unreliable
- ❌ Requires sudo

### Don't use for

- Multi-GPU DDP / FSDP (cuda-checkpoint is single-process limited)
- Cross-machine migration (use transformers ckpt + RESUME instead)
- Critical production training in DISK mode without smoke test first

## Claude Code skill

This repo is itself a [Claude Code](https://claude.com/claude-code) skill. Install:

```bash
git clone https://github.com/dark4scope/gpu-pause.git ~/.claude/skills/gpu-pause
chmod +x ~/.claude/skills/gpu-pause/scripts/gpu-pause.sh
```

Then in Claude Code, just say "pause PID 12345 training to free up the GPU" and Claude calls this skill, runs `freeze`, and tells you the restore command.

## License

MIT — see [LICENSE](LICENSE)

## Credits

- [NVIDIA cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint) — core GPU state freeze
- [CRIU](https://criu.org) — process-level checkpoint/restore

## Links

- Chinese README: [README.md](README.md)
