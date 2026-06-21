# gpu-pause

> **冻结正在跑的 NVIDIA GPU 训练进程,让出 GPU 显存,之后字节级无损恢复继续训练**

基于 NVIDIA 官方 [`cuda-checkpoint`](https://github.com/NVIDIA/cuda-checkpoint) (2024.4 ship) + 可选 [CRIU](https://criu.org) 进程级 checkpoint,把任意 GPU 进程冻结让出 GPU,之后**字节级无损**恢复。已实测 model 参数 hash / optimizer state hash / forward 输出完全一致。

---

## 使用场景

- 训练跑到一半,临时想用 GPU 跑别的 / 让别人用 / 跑评估 → **freeze**
- 训练过夜想关机省电 → **dump** 到磁盘,明天 **undump** 继续
- 不想中断训练但需要释放显存做小任务 → 单卡轮流跑

## 工作原理

```
┌─────────────────────────────────────────────────────────────────┐
│                          训练进程 (PID)                          │
│                                                                  │
│  Python ─→ PyTorch ─→ CUDA Runtime ─→ Driver ─→ GPU 显存 36 GB  │
└─────────────────────────────────────────────────────────────────┘

  ↓ cuda-checkpoint --action=checkpoint   (RAM mode)

┌─────────────────────────────────────────────────────────────────┐
│                          训练进程 (PID)                          │
│  Python ─→ PyTorch ─→ [GPU state 搬到 CPU RAM, 显存释放]         │
└─────────────────────────────────────────────────────────────────┘
                          ↓
                  ⚡ GPU 显存 = 0 MiB, 别人可用 ⚡

  ↓ criu dump                            (DISK mode 进一步)

┌─────────────────────────────────────────────────────────────────┐
│              磁盘 dump 文件 1.2 GB ~ 50 GB                       │
│   /tmp/gpu-pause-dumps/pid-12345-20260621-2230/                  │
│   ├── pages-*.img   (进程 RAM + GPU 内容)                        │
│   ├── core-*.img    (CPU register / thread state)                │
│   └── ... (43 个文件)                                            │
└─────────────────────────────────────────────────────────────────┘
                          ↓
              ⚡ GPU + RAM 都释放, 机器可重启 ⚡

  ↓ criu restore (+ cuda-checkpoint restore 自动)

恢复后 PID 仍是原 PID, GPU 自动 re-attach, 进程从断点字节级无损继续
```

## 安装

### 1. cuda-checkpoint(必装)

```bash
# 必须 NVIDIA driver R535+ (CUDA 12.4+)
nvidia-smi | head -3

# 下载 binary (静态链接, 跟当前 driver 版本自动对齐)
curl -sL https://github.com/NVIDIA/cuda-checkpoint/raw/main/bin/x86_64_Linux/cuda-checkpoint -o /tmp/cuda-checkpoint
sudo install -m 755 /tmp/cuda-checkpoint /usr/bin/cuda-checkpoint
cuda-checkpoint --help | head -2
```

### 2. CRIU(只在 DISK mode 用)

Ubuntu 22.04 / 24.04 系列需要走 PPA(系统 apt 包已 obsoleted):

```bash
sudo add-apt-repository -y ppa:criu/ppa
sudo apt-get update && sudo apt-get install -y criu
sudo criu check  # 必须输出 "Looks good"
```

### 3. 本仓库脚本

```bash
git clone https://github.com/dark4scope/gpu-pause.git
cd gpu-pause
sudo install -m 755 scripts/gpu-pause.sh /usr/local/bin/gpu-pause
gpu-pause help
```

或者一键脚本:

```bash
curl -sL https://raw.githubusercontent.com/dark4scope/gpu-pause/main/install.sh | bash
```

## 用法

```
gpu-pause <action> [args]

  status PID        查状态 (running / checkpointed)
  freeze PID        RAM mode: GPU 释放, 进程睡 RAM
  restore PID       RAM mode: GPU 重 attach, 进程继续
  dump   PID [DIR]  DISK mode: 完整 dump, PID killed, GPU+RAM 都释放
                    DIR 默认 /tmp/gpu-pause-dumps/pid-<PID>-<timestamp>
  undump DIR        DISK mode: criu restore, PID 自动 reuse, GPU 自动 attach
  list              列出所有 dump
  help              显示 usage
```

### 例 1:RAM mode(最常用)

```bash
# 找到训练 PID
ps -ef | grep python | grep train

# 冻
$ gpu-pause freeze 12345
[OK] PID 12345  running -> checkpointed
     GPU mem: 36251 MiB -> 1 MiB
     restore: bash gpu-pause restore 12345

# 此时 GPU 空了, 可以跑别的训练 / 推理 / 评估 ...

# 恢
$ gpu-pause restore 12345
[OK] PID 12345  checkpointed -> running
     GPU mem: 1 MiB -> 36254 MiB
```

进程从冻结时的 step 继续,**model 参数 / optimizer m+v / RNG state 全部字节级一致**,跟从未冻过完全没区别。

### 例 2:DISK mode(关机 hibernate)

```bash
# 完整 dump
$ gpu-pause dump 12345
[dump] PID=12345  ->  /tmp/gpu-pause-dumps/pid-12345-20260621-2230
1 MiB
[OK] PID killed
[done] dump 1.2G in 49 files at /tmp/gpu-pause-dumps/pid-12345-20260621-2230
1 MiB
[restore later] bash gpu-pause undump /tmp/gpu-pause-dumps/pid-12345-20260621-2230

# 此时进程死了, GPU + RAM 都释放, 可以重启机器

# 恢复
$ gpu-pause undump /tmp/gpu-pause-dumps/pid-12345-20260621-2230
[restore] /tmp/gpu-pause-dumps/pid-12345-20260621-2230
[OK] PID 12345 restored
12345 SNl  00:03 /path/to/python train.py
36254 MiB
```

注意 PID 仍是 12345(CRIU 默认 reuse 原 PID 如果可用)。

## 实测验证(本仓库自带 smoke 脚本)

`examples/verify_byte_exact.py` 跑一个 PyTorch 小模型 + AdamW 20 step → 进入 forward loop 维持 GPU activity,**自动验证**冻结/恢复后:

- ✅ `model.parameters` hash 完全一致
- ✅ `optimizer.state` hash 完全一致
- ✅ 同一固定 input forward 输出字节级相同
- ✅ counter tensor 在 freeze 时 = N,restore 后从 N+1 继续

实测时间线(本机 3090 + driver R595):

| T | 事件 | GPU mem | counter | PID 状态 |
|---|---|---|---|---|
| T0 | 正常跑 | 512 MiB | 27 | running |
| T+1s | `gpu-pause freeze` | **1 MiB** | 27 | **checkpointed** |
| T+20s | 等待中,GPU 空 | 1 MiB | 27 | sleep in RAM |
| T+21s | `gpu-pause restore` | 512 MiB | 27 | running |
| T+22s | 继续跑 | 153 MiB | **28** | running |
| T+23s | | | 29 | |

**model_hash / optim_hash / forward_output_hash 在 freeze 前后完全一致**。

## 限制 / 风险

### RAM mode

- ✅ **本仓库已实测**字节级无损 (single-threaded PyTorch + Linear + AdamW)
- ✅ **SFTTrainer + unsloth + flash-attn + DL_WORKERS=4 + grad_ckpt 实测 work**(2026-06-21 本机 0.8B Qwen3.5 SFT,GPU 4438 MiB → 4 MiB,restore 后训练无缝继续 step 86→117)
- ❌ 进程 RAM = 原 RAM + GPU mem(9B SFT 需 +36 GB RAM)
- ❌ 长 sleep 数小时+可能踩 kernel reaper / heartbeat

### DISK mode

- ✅ 已实测 single-threaded Python+PyTorch byte-exact
- ✅ GPU + RAM 都释放,机器可重启(同机同 mount/namespace 内)
- ❌ **DataLoader(num_workers > 0) 实测 fail**(2026-06-21):PyTorch DataLoader **子进程**(`pt_data_worker`)持有 CUDA mapping(`0x200200000` device memory range),CRIU 的 `cuda_plugin` 不能 dump non-regular mapping → `Dumping FAILED`
- ✅ **DataLoader(num_workers=0) 实测 work**(2026-06-21):单进程 dataloader 无子进程,dump 1.1 GB / 49 files,PID killed,GPU 释放,restore 后 PID reuse + GPU auto-attach + step 连续(475 → 500 → 525...)+ loss 仍稳定
- ⚠️ Workaround:把 `DataLoader(num_workers=0)` — 牺牲 ~30% 数据加载速度换 dump 兼容
- ❌ Dump 大小 = 进程 RAM + GPU 显存(9B 模型 ~40-50 GB)
- ❌ 跨机器 restore 不可靠
- ❌ 需要 sudo

### 不要用本工具的场景

- Multi-GPU DDP / FSDP 训练(cuda-checkpoint 是单进程限制)
- 跨机器迁移(走 transformers ckpt + RESUME 更稳)
- 关键生产训练用 DISK mode 前必先 smoke

## Claude Code Skill 集成

本仓库本身就是 [Claude Code](https://claude.com/claude-code) skill,装到 `~/.claude/skills/gpu-pause/` 后,Claude 看到「冻结 / 暂停训练 / 让出 GPU / hibernate」等关键词会自动调用:

```bash
git clone https://github.com/dark4scope/gpu-pause.git ~/.claude/skills/gpu-pause
chmod +x ~/.claude/skills/gpu-pause/scripts/gpu-pause.sh
```

之后在 Claude Code 里直接说:"暂停 PID 12345 的训练让出 GPU",Claude 会调本 skill 跑 `freeze`,完成后告诉你恢复命令。

## License

MIT — 见 [LICENSE](LICENSE)

## 致谢

- [NVIDIA cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint) — 核心 GPU state 冻结
- [CRIU](https://criu.org) — 进程级 checkpoint/restore

## 相关链接

- 英文 README: [README.en.md](README.en.md)
- 详细 learning(开发背景 + 实测 KPI 表)在 `examples/` 目录
