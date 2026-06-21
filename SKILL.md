---
name: gpu-pause
description: >
  Freeze and restore NVIDIA GPU training processes using cuda-checkpoint (RAM mode) or
  cuda-checkpoint + CRIU (DISK mode). Use when user asks to: pause/freeze GPU training,
  release GPU 显存 给别人用, suspend training, hibernate training to disk, restore frozen
  training, 冻结/暂停 GPU 训练, GPU 训练 pause+resume, 训练让出 GPU. Two modes:
  RAM (process keeps RAM, GPU released, ~1s) or DISK (CRIU dump, process killed, full hibernate).
  Verified byte-exact (model+optimizer hash identical) on 2026-06-21 with driver R595 + CUDA 13.0.
---

# GPU Pause: 冻结 + 恢复 GPU 训练进程

把任意 NVIDIA GPU 进程冻结让出 GPU,之后**字节级无损**恢复继续。底层用 NVIDIA `cuda-checkpoint`(2024.4 ship)+ 可选 CRIU(磁盘 dump)。

## 两种 mode

| Mode | 适用 | GPU 释放 | 进程仍活 | 恢复时长 | 占用 |
|---|---|---|---|---|---|
| **RAM** (cuda-checkpoint 单工具) | 暂停 < 数小时,同机 | ✅ | ✅ (sleep 在 RAM) | ~1s | GPU mem 全搬 RAM (9B 训练 +36 GB RAM) |
| **DISK** (cuda-checkpoint + CRIU) | 过夜 / 关机 / 释放 RAM | ✅ | ❌ killed,dump 到磁盘 | ~3-5s + 写盘 30-60s | 磁盘 ~ 进程总 RAM (9B ≈ 40-50 GB) |

## 用法

```bash
bash /home/xgwang/.claude/skills/gpu-pause/scripts/gpu-pause.sh <action> [args]
```

| Sub-cmd | Args | 行为 |
|---|---|---|
| `status` | `PID` | 查 state(`running` / `checkpointed`)|
| `freeze` | `PID` | **RAM mode** GPU 释放,进程睡 RAM |
| `restore` | `PID` | **RAM mode** GPU 重 attach,进程继续 |
| `dump` | `PID [DIR]` | **DISK mode** 完整 dump 到 DIR(默认 `/tmp/gpu-pause-dumps/pid-<PID>-<ts>`),PID killed |
| `undump` | `DIR` | **DISK mode** criu restore(CRIU 自动 reuse 原 PID + GPU auto re-attach,无需 cuda-checkpoint restore) |
| `list` | | 列默认 root 下所有 dump |
| `help` | | usage |

## 触发用户意图(Claude 调用规则)

用户说类似下面任何一句,都该用本 skill:
- "冻结 / 暂停 / pause / freeze [GPU / 训练]"
- "让出 GPU 给 X 用"
- "dump 训练到磁盘 / hibernate"
- "resume / 恢复 / restore 之前冻结的训练"
- "释放 GPU 显存(进程暂时不要)"

**Claude 操作流程**:
1. 找出目标 PID(如 user 说"暂停训练",先 `ps -ef | grep <训练入口>` 找 PID,或问 user)
2. 若 RAM 充足 + 短暂停 → `freeze`;若过夜 / 关机 → `dump`
3. 跑 sub-command,验证 `nvidia-smi --query-gpu=memory.used` 确认释放
4. 报告用户 + 给出对应 restore 命令

## 例子

```bash
# RAM mode (最常用)
bash gpu-pause.sh status 12345          # 查
bash gpu-pause.sh freeze 12345          # 冻 (GPU 释放, 进程 RAM 睡)
# ... 别人用 GPU ...
bash gpu-pause.sh restore 12345         # 恢复, 字节级无损继续

# DISK mode (彻底 hibernate)
bash gpu-pause.sh dump 12345            # PID killed, ~40 GB 落盘
# ... 关机 / 重启 / 几天后 ...
bash gpu-pause.sh undump /tmp/gpu-pause-dumps/pid-12345-20260621-2230
```

## 限制 / 风险

### RAM mode
- ✅ 已实测**字节级无损**(model param hash + optimizer state hash + forward output 完全一致)
- ❌ 进程 RAM = 原 RAM + GPU mem swap(本机 62 GB RAM 跑 9B 训练 + freeze 紧张)
- ❌ 长 sleep(数小时+)可能踩 kernel reaper / heartbeat
- ⚠️ **没验证**:SFTTrainer + 12 dataloader workers + flash-attn + torch.compile(用前 smoke)

### DISK mode
- ✅ GPU + RAM 都释放,机器可重启(但 CRIU dump 绑定 mount points / namespaces,同机同 env)
- ✅ 已实测 single-threaded Python+PyTorch byte-exact
- ❌ **没验证**多线程 Python(SFTTrainer 12 dataloader workers + flash-attn — CRIU 对 multi-threaded 出名脆弱)
- ❌ Dump 大小 = 进程 RAM + GPU mem(9B 模型 ~40-50 GB)
- ❌ 需要 sudo(criu)
- ❌ 跨机 restore 不可靠

## 安装前置

```bash
# cuda-checkpoint (始终需要)
env http_proxy=http://127.0.0.1:17890 https_proxy=http://127.0.0.1:17890 \
  curl -sL https://github.com/NVIDIA/cuda-checkpoint/raw/main/bin/x86_64_Linux/cuda-checkpoint -o /tmp/cc
sudo install -m 755 /tmp/cc /usr/bin/cuda-checkpoint
nvidia-smi  # driver R535+

# CRIU (只在 dump/undump 用)
sudo add-apt-repository -y ppa:criu/ppa && sudo apt-get update
sudo apt-get install -y criu
sudo criu check  # 必须 "Looks good"
```

## 何时**不要**用本 skill

- Multi-GPU DDP/FSDP(cuda-checkpoint 单进程限制)
- 跨机器迁移(走 transformers ckpt + RESUME 更稳)
- 生产 9B SFT 关键训练用 DISK mode 前必先 smoke(多线程 + CRIU 风险)

## 详细 learning

完整背景 / 实测 KPI / SOP 沉淀:`~/.claude/.learnings/2026-06-21_cuda-checkpoint-gpu-freeze-resume.md`
