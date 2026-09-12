# 龙芯 3A6000 + RX 7900 XTX + ROCm 7.2：PyTorch / Triton / vLLM 从零构建全记录

本文记录在这块板子上从零把 PyTorch、Triton、vLLM 跑起来、并用 Qwen 模型做推理的**全部过程**，
包括每一个踩过的坑、原因、修法，以及**当前仍未解决的那个硬问题**。

---

## 0. 实测基线（先对一遍，版本不对后面全错）

| 项 | 实测值 | 说明 |
|---|---|---|
| CPU | Loongson 3A6000, LoongArch64, 8 核 | |
| 内存 | 31 GB | |
| GPU | AMD Radeon RX 7900 XTX (gfx1100) | VRAM 23.98 GiB (25,753,026,560 B) |
| OS | Arch Linux (loong64) | |
| Python | 3.14.7 | 系统 python，PEP 668 生效 |
| PyTorch | 2.13.0，`torch.version.hip=7.2.26063` | 来自 Arch 包 `python-pytorch-opt-rocm 2.13.0-6` |
| ROCm | 7.2.0 (`/opt/rocm`) | 包：hip-runtime-amd / hipblas / miopen-hip / hipblaslt … |
| 系统 LLVM/clang | 22.1.8 | **不能给 triton 3.8 用** |
| 自建 LLVM | `/opt/llvm-pin` = 24.0.0git | triton 3.8 专用 |
| Triton | 3.8.0（`/root/triton`，fork `BoneInscri/triton`） | **当前是问题根源，见 §5、§11** |
| vLLM | `0.1.dev21041+gd2906091b.d20260909`（`/root/vllm`） | editable 安装 |
| 磁盘 | `/` 393 GB（用 270 / 余 104） | `/tmp` 是 **16 GB tmpfs**，见 §2.1 |

**最重要的一个前提**：PyPI 上**没有 loongarch64 的 torch / triton / vllm 轮子**。
凡是需要这些东西的包，要么用发行版包，要么源码编译——没有第三条路。

---

## 1. 三条全局铁律（先读这个，能省掉 80% 的诡异报错）

### 铁律一：`/tmp` 会被写满，所有命令都带 `TMPDIR=/var/tmp`

`/tmp` 是 16 GB tmpfs。板子上有个进程在**无限**往里写日志：

```
root  104417  1  97  13:01 pts/1  06:11:34  /root/xsched/output/bin/xserver HPF 50000
   → 持续写 /tmp/dyn-xs.log（表观大小 164~185 GB 的稀疏文件，实际十几分钟填满 16 GB）
```

tmpfs 一满，症状是**假性磁盘满**，而且报错位置离真因很远：

```
LLVM ERROR: IO failure on output stream: No space left on device
```
（或各种"编译到一半莫名失败"）

规矩：

```bash
df -h /tmp                     # 先看一眼
truncate -s 0 /tmp/dyn-xs.log  # 满了就清
export TMPDIR=/var/tmp         # 所有编译/运行命令都带（/ 根分区有 100+ GB）
```

根治办法（二选一）：
- 杀掉那个 xserver：`kill 104417`（**注意**：它可能是你的图形会话，先确认）；
- 板上**没有 crontab**，用 systemd timer 每分钟 truncate：
  ```ini
  # /etc/systemd/system/trim-dyn-xs.service
  [Service]
  Type=oneshot
  ExecStart=/usr/bin/truncate -s 0 /tmp/dyn-xs.log
  # /etc/systemd/system/trim-dyn-xs.timer
  [Timer]
  OnCalendar=*:0/1
  [Install]
  WantedBy=timers.target
  ```
  然后 `systemctl enable --now trim-dyn-xs.timer`。

> 注意：**不能**靠 `mv` 或做 symlink 解决。那个进程的文件描述符已经打开在旧 inode 上，
> 换路径只影响"新打开"的进程，它照样把 tmpfs 写满。

### 铁律二：不要在 `/root`、`/root/vllm`、`/root/triton`、`/root/sglang` 里运行 python

这些目录下有同名的源码目录（`/root/vllm/vllm`、`/root/triton/python/triton`），
在它们里面启动 python 时 `sys.path[0]` 会指向该目录，`import vllm` 就解析成一个**命名空间包**，
报的是这种看着莫名其妙的错：

```
unknown location
cannot import name '__version__' from 'vllm'
```

规矩：**cd /tmp 再跑**；自己写的脚本放 `/opt/tests/`，别放 `/root` 下。

### 铁律三：不要用 pip 装 torch / triton；PEP 668 要加 `--break-system-packages`

```bash
pip install torch               # ❌ 会把能用的 ROCm 版 torch 覆盖掉（而且 PyPI 上根本没有龙芯版）
pip install -e .                # ❌ 报 externally-managed-environment
python3 -m pip install -e . --no-build-isolation --no-deps --break-system-packages   # ✅
```

**永远用 `python3 -m pip`**，不要裸 `pip`（这台机器上多个 python 前缀，很容易装错地方）。

---

## 2. 阶段 0：系统层准备

### 2.1 处理 tmpfs（见铁律一）

### 2.2 目录遮蔽（见铁律二）

### 2.3 时钟偏移

板子 RTC 与文件 mtime 不同步会让构建系统误判"文件已是最新"而**跳过安装**。
典型现象：`ninja install` / `cmake --install` 只装了零星几个文件，`lib/cmake/llvm` 都不存在。
处置：`rm -rf <prefix>` 之后重新 `cmake --install <build> --prefix <prefix>`，或用 `touch` 刷新时间戳。

### 2.4 常用环境变量（后面反复出现，先记住）

| 变量 | 值 | 用途 |
|---|---|---|
| `TMPDIR` | `/var/tmp` | 避开 tmpfs 满 |
| `ROCM_HOME` | `/opt/rocm` | vLLM HIP 编译 |
| `PYTORCH_ROCM_ARCH` | `gfx1100` | 指定 GPU 架构 |
| `LLVM_SYSPATH` | `/opt/llvm-pin` | triton 用它找 LLVM |
| `JSON_SYSPATH` | `/usr` | triton 编 nvidia backend 需要系统 nlohmann-json |
| `TRITON_OFFLINE_BUILD` | `1` | 不联网下载 |
| `TRITON_KERNELS_SRC_DIR` | `/root/triton_kernels_src` | vLLM `.deps` 用 |

---

## 3. 阶段 1：PyTorch（"PyTorch 无法使用"的真相）

### 3.1 为什么 pip 装不上

```
pip install torch
ERROR: Could not find a version that satisfies the requirement torch (from versions: none)
```
PyPI 没有 loongarch64 wheel。网上所有 `pip install torch --index-url .../rocm7.x` 的教程在这台机器上**全部无效**。

### 3.2 正确做法：用 Arch 的 ROCm 版 torch，不要 pip

```bash
pacman -S python-pytorch-opt-rocm      # 实测 2.13.0-6，torch.version.hip = 7.2.26063
# 不要装 python-pytorch（那是 CPU 版，装完 torch.cuda.is_available() 是 False）
```

选包要点：**必须带 `-opt-rocm`**（为 gfx1100 优化过的 ROCm 构建）。

### 3.3 验证（这一段必须过，不过就别往下走）

```bash
cd /tmp && python3 -c "
import torch
print('torch', torch.__version__, '| hip', torch.version.hip)
print('cuda_available', torch.cuda.is_available(), '| count', torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/2**30, 2), 'GiB')
"
# 期望：2.13.0 | hip 7.2.26063 / True / 1 / AMD Radeon RX 7900 XTX 23.98 GiB
```

### 3.4 "torch 用不了"排查清单

1. `torch.__file__` 指向 `/usr/lib/python3.14/site-packages/torch/...` 才对；
   若指向 `~/.local/...` 或别的 venv，说明装了第二份 torch，先把那份删掉。
2. `torch.version.hip` 是 `None` → 装成了 CPU 版（`python-pytorch`），换 `python-pytorch-opt-rocm`。
3. `torch.cuda.device_count()==0` → 用 `rocminfo | grep gfx` 确认 ROCm 是否认得卡；确认当前用户在 `video`/`render` 组。
4. gfx1100 是 ROCm 7.2 原生支持的架构，**不需要** `HSA_OVERRIDE_GFX_VERSION`（乱设反而会跑错 kernel）。
5. 一旦发现"GPU 能用但结果不对"，先怀疑**编译器优化级别**，见 §6.3。

---

## 4. 阶段 2：自建 LLVM 24（triton 的前置，最贵的一步）

### 4.1 为什么必须自建

triton 3.8 需要 **LLVM 24 + MLIR + LLD**。板上系统 LLVM 是 22.1.8，**版本不兼容**，典型报错是一堆
C++ API 参数个数对不上：

```
'PropertyRef' / 'RegionSuccessor' / 'Resource' … 2 arguments
```

（我们后来确实用 LLVM 22 试过，就是这些错。**别在 LLVM 22 上浪费时间**。）

### 4.2 取源码

pin 来自 triton 树的 `cmake/llvm-info.json`（该文件给出所需 llvm-project 提交与预编译包 sha256）。
板上自建时用的是：

```
triton-lang/llvm-project @ 1df311a6f7d8af05d068ce835b60dfa5b199d2f8   （记录在 cmake/llvm-build-info.json）
注意：树里 llvm-info.json pin 的是 b010a18d2b648cab83c83967ff26b8fde11acdc6，两者都是 LLVM 24，但 API 细节有差
      → 用 1df311a6 时需要对 triton 源码打 1 行补丁（见 §5.2）
```

### 4.3 实测可用的 cmake 配置

```bash
cmake -G Ninja -S <llvm-project-src> -B /opt/src/llvm-pin-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir;lld" \
  -DLLVM_TARGETS_TO_BUILD="AMDGPU;NVPTX" \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DCMAKE_INSTALL_PREFIX=/opt/llvm-pin
ninja -C /opt/src/llvm-pin-build -j8
```

要点：
- 只要 `AMDGPU`，但**必须**带 `NVPTX`——triton 的 nvidia backend 在配置阶段就会找 `LLVMNVPTX*` 库，缺了就 configure 失败。
- 必须带 `mlir`（triton 编译内核全靠它）和 `lld`。
- `LLVM_INSTALL_UTILS=ON`（triton 需要 `llvm-config` 等工具）。
- 8 核 Loongson 上一次完整构建是**小时级**，放后台跑。

### 4.4 安装坑（时钟/mtime）

如果 `ninja install` 之后 `lib/cmake/llvm` 都不存在，就是 §2.3 的 mtime 问题：

```bash
rm -rf /opt/llvm-pin
cmake --install /opt/src/llvm-pin-build --prefix /opt/llvm-pin
ls /opt/llvm-pin/lib/cmake/{llvm,mlir,lld}   # 三个都要在
/opt/llvm-pin/bin/llvm-config --version      # 24.0.0git
```

---

## 5. 阶段 3：Triton（**最关键的分叉点，选错版本后面全崩**）

### 5.1 版本选择：3.5.1 还是 3.8？（实测证据）

板上现在装的是 **triton 3.8.0（fork 的 main 分支）**，而这是**后续所有 GDN 模型崩溃的根源**。原因是：

```
triton 提交 9debefe44  [BC breaking][FRONTEND] Remove block pointer support (#10833)
   → tl.make_block_ptr / tl.advance 被彻底删除（前端 + MLIR op + lowering 一起删）
   → 现在的 HEAD 在这个提交之后 389 个提交
```

而 vLLM 里**仍有 323 处 `tl.make_block_ptr`**（其中 121 处在 vendored flash-linear-attention）。
只要模型带线性注意力（GatedDeltaNet），JIT 编译时必炸，见 §11。

**正确配对是什么？** ROCm 官方的 triton wheel 索引里最高只到 **3.5.1**（含 cp314）：

```
https://download.pytorch.org/whl/rocm7.2/pytorch-triton-rocm/
  ... 3.4.0, 3.5.0, 3.5.1     ← 最高 3.5.1
```

也就是说 **ROCm 7.2 + torch 2.13 配的是 triton 3.5.1**，不是 3.8。3.5.1 仍然有 block pointer。

**结论 / 建议**：从你的 fork 的 `v3.5.1` tag 编 triton，而不是 main。代价是 3.5.1 的 LLVM pin 与 3.8 不同，
很可能要**再编一次 LLVM**（又是小时级），换来的是 **vLLM 一行都不用改**、323 处 kernel 全部可用。

### 5.2 如果坚持用 3.8：一个必须的 1 行补丁

`python/src/llvm.cc`（fork 里唯一改动，`git status` 显示 `M python/src/llvm.cc`）：

```diff
- [kVectorCombinePassName](StringRef passName, Any)
+ [kVectorCombinePassName](StringRef passName, llvm::IRUnitRef)
```

### 5.3 实测可用的编译命令

```bash
cd /root/triton
TMPDIR=/var/tmp \
TRITON_OFFLINE_BUILD=1 \
LLVM_SYSPATH=/opt/llvm-pin \
JSON_SYSPATH=/usr \
TRITON_APPEND_CMAKE_ARGS="-DLLVM_DIR=/opt/llvm-pin/lib/cmake/llvm -DMLIR_DIR=/opt/llvm-pin/lib/cmake/mlir -DLLD_DIR=/opt/llvm-pin/lib/cmake/lld -DTRITON_BUILD_PROTON=OFF" \
PATH=/opt/llvm-pin/bin:/usr/bin:$PATH \
python3 -m pip install -e . --no-build-isolation --break-system-packages
```

各参数为什么需要：

| 参数 | 不加会怎样 |
|---|---|
| `LLVM_SYSPATH` / `-DLLVM_DIR -DMLIR_DIR -DLLD_DIR` | 找不到 MLIR / `find_package(LLD)` 失败 |
| `JSON_SYSPATH=/usr` | nvidia backend 的 gsan 缺 nlohmann-json |
| `-DTRITON_BUILD_PROTON=OFF` | Proton 需要 `cuda.h`，板子上没有 → 直接失败 |
| `TRITON_OFFLINE_BUILD=1` | configure 阶段会去 clone googletest 而卡死 |
| `TMPDIR=/var/tmp` | 编到一半报 `LLVM ERROR: IO failure … No space left on device` |
| `PATH=/opt/llvm-pin/bin:...` | 用的是系统 clang 22，和 LLVM 24 头文件打架 |

另外要补一个软链接（nvidia backend 的 gsan 阶段写死找 `clang++`）：

```bash
ln -sf /usr/bin/clang++ /opt/llvm-pin/bin/clang++
```

### 5.4 验证

```bash
cd /tmp && python3 /opt/tests/triton_gpu_test.py     # 期望：TRITON KERNEL OK, max_err = 0.0
```

**必须验证到数值正确**，不能只看 `import triton` 成功。

---

## 6. 阶段 4：vLLM 源码编译

### 6.1 绝对不要用预编译 wheel

```bash
VLLM_USE_PRECOMPILED=1 python3 setup.py develop
# ValueError: No compatible wheel found for loongarch64 at https://pypi.amd.com/vllm-rocm/simple/vllm/
```
`VLLM_USE_PRECOMPILED` 的意思是"去下官方预编译轮子"——龙芯没有。**必须源码编译，不要设这个变量。**

### 6.2 依赖准备

```bash
cd /tmp && TMPDIR=/var/tmp python3 -m pip install --break-system-packages \
  cmake ninja hatch-fancy-pypi-readme scikit-build-core
```
缺 `hatch-fancy-pypi-readme` / `scikit-build-core` / `ninja` 的表现是 `metadata-generation-failed`。
Arch 上全部要 `--break-system-packages`。

### 6.3 两个必须的本地补丁（否则编出来"能跑但算错/直接崩"）

文件：`/root/vllm/cmake/utils.cmake`，`GPU_FLAGS`（HIP 分支）：

```diff
- "-Werror=unused-variable"
+ "-Wno-error=unused-variable"
+ "-O2"
```

两个补丁各自解决什么：

| 补丁 | 不加的症状 |
|---|---|
| `-Wno-error=unused-variable` | ROCm 7.2 的 HIP 头文件里有 `unused variable 'lastLane'`，被 `-Werror` 当错误 → 编译直接失败 |
| `-O2` | 默认 `-O0` 时 LLVM 在 gfx1100 上生成非法的 DPP 控制字 → 运行时报 `Illegal instruction detected: Invalid dpp_ctrl value ... GFX10+`（**这个坑极难查**，见 llama.cpp issue #18396） |

### 6.4 编译（按记录整理，配合两块前置）

前置：把 ROCm/triton 里的 `triton_kernels` 源码目录准备好（vLLM 的 `.deps` 构建要用）：

```bash
# ROCm/triton @ 0f380657… 的 python/triton_kernels/triton_kernels 共 41 个 .py
export TRITON_KERNELS_SRC_DIR=/root/triton_kernels_src
```

编译：

```bash
cd /root/vllm
export TMPDIR=/var/tmp
export ROCM_HOME=/opt/rocm
export PYTORCH_ROCM_ARCH=gfx1100
export TRITON_KERNELS_SRC_DIR=/root/triton_kernels_src
python3 -m pip install -e . --no-build-isolation --no-deps --break-system-packages
```

产物：`0.1.dev21041+gd2906091b.d20260909.rocm720`（editable，源码树在 `/root/vllm`）。

### 6.5 运行期依赖补齐（编完还要装一堆）

导入链会一个个报缺，逐个补（都要 `--no-build-isolation --no-deps --break-system-packages`，
Arch + loongarch 上很多包没有 wheel）：

| 报错 | 处置 |
|---|---|
| `ImportError: cannot import name 'NamespaceTool' from 'openai.types.responses'` | 升 `openai` 到 3.10.0（连带 `httpx2` / `httpcore2` / `truststore`） |
| `No module named 'xgrammar'` | 装 xgrammar（依赖下面那个 tvm-ffi） |
| `No module named 'tvm_ffi'` | `apache-tvm-ffi`（`xgrammar` 的依赖） |
| `No module named 'compressed_tensors'` / `'einops'` | 直接装 |
| `No module named 'blake3'` | 多模态哈希用；装 `blake3`（1.0.9 已装） |
| `No module named 'sentencepiece'` / `'dill'` | 装（sentencepiece sdist 走 cmake 编译） |

### 6.6 验证：**必须跑真推理，不能只看 import**

这是本项目**最重要的方法论教训**（见 §10.2）：

```bash
cd /tmp && TMPDIR=/var/tmp python3 /opt/tests/vllm_smoke.py
```
`/opt/tests/vllm_smoke.py` = 加载 0.5B → 两条 prompt → 打印答案和 tok/s。
实测通过输出：

```
[load] 21.2s
Q: 用一句话介绍你自己
A: 我，Qwen，是阿里巴巴云研发的超大规模语言模型…
   [27 tok in 1.0s = 25.86 tok/s]
Q: 1+1=?只回答数字
A: 1+1=2
```

---

## 7. 阶段 5：模型选择与显存计算

### 7.1 显存数学（24 GB 很紧，先算再下）

**权重**：必须量化。Qwen3.8-27B 的 bf16 是 18 个分片 ≈ 54 GB，放不下；
INT4/AWQ 量化后 19~20 GB（实测加载 17.64 GiB）。

**KV cache**（Qwen3.8-27B：64 层、4 个 KV head、head_dim 256）：

```
每 token = 2(K,V) × 4 head × 256 dim × 2 B(bf16) × 64 层 = 256 KB/token
max_model_len=2048 → 0.5 GiB
max_model_len=4096 → 1.0 GiB
```

所以 2048 上下文时：权重 ~18 GiB + KV 0.5 GiB + 激活 ~0.3 GiB ≈ 19 GiB，
`gpu_memory_utilization=0.95`、`--enforce-eager` 是合适的组合（0.5 会因装不下权重而失败）。

### 7.2 量化格式

实测可用：`RedHatAI/Qwen3.8-27B-INT4`（compressed-tensors W4A16 g128）
→ vLLM 打印 `Using RDNA3W4A16LinearKernel for CompressedTensorsWNA16`，说明 gfx1100 有对应 kernel。
其它候选：`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`、`gratex/Qwen3.8-27B-W4A16-g128-sym-GPTQ`。
**注意**：这份 vLLM 构建**没有 GGUF loader**，GGUF（含 ollama 用的那份）不能在 vLLM 里跑。

### 7.3 下载

板子直连 GitHub/HF 只有 **30~550 B/s**，基本不可用。做法：用有快网的中转机
（实测 HF-mirror ~5.8 MB/s，ModelScope ~12 MB/s）下好，再 `rsync`/`scp` 到板子 `/root/models/`。

### 7.4 **架构兼容性**（选模型前必看，§11 的坑就在这里）

| 模型架构 | vLLM 里能不能跑 | 原因 |
|---|---|---|
| `Qwen2ForCausalLM`（Qwen2.5-0.5B） | ✅ 已实测 | 纯 full attention，不碰线性注意力 kernel |
| `Qwen3_5ForConditionalGeneration`（Qwen3.8-24B/27B） | ❌ 当前不可用 | 混合线性注意力（GatedDeltaNet），GPU 编译需要 `tl.make_block_ptr`，而 triton 3.8 已删除该 API |
| 其它带线性注意力的：`kimi_k3` / `glm5next` / `minimax_m3` | ❌ 同理 | 这些模型的 ops 目录里也全是 block pointer |

**避坑结论**：在当前 triton 3.8 环境下，**只能跑纯 dense / full-attention 的模型**。

---

## 8. 阶段 6：跑推理

### 8.1 脚本模板（`if __name__ == "__main__":` 不能省）

vLLM v1 引擎用 `spawn` 起子进程。没有 main guard 会报：

```
An attempt has been made to start a new process before the current process
has finished its bootstrapping phase
```

```python
# /opt/tests/vllm_loop.py
from vllm import LLM, SamplingParams

MODEL = "/root/models/Qwen2.5-0.5B-Instruct"

def main():
    llm = LLM(model=MODEL, enforce_eager=True,
              gpu_memory_utilization=0.5, max_model_len=2048)
    sp = SamplingParams(temperature=0.7, max_tokens=128)
    history = []
    while True:
        q = input("\n你> ").strip()
        if q in ("exit", "quit", ""):
            break
        history.append({"role": "user", "content": q})
        out = llm.chat(history, sp)
        a = out[0].outputs[0].text
        history.append({"role": "assistant", "content": a})
        print("模型>", a)
        if len(history) > 12:          # 防超 max_model_len
            history = history[-12:]

if __name__ == "__main__":
    main()
```

### 8.2 运行

```bash
cd /tmp && TMPDIR=/var/tmp python3 /opt/tests/vllm_loop.py
```
（`cd /tmp` 是铁律二；`TMPDIR` 是铁律一。）

### 8.3 参数速查

| 参数 | 建议 | 说明 |
|---|---|---|
| `enforce_eager=True` | 建议开 | 关掉 torch.compile / CUDAGraph，ROCm 上更稳 |
| `gpu_memory_utilization` | 小模型 0.5；19 GB 级模型 0.95 | 太小装不下权重 → OOM |
| `max_model_len` | 2048 | 见 §7.1 的 KV 计算 |
| `language_model_only=True` | 多模态模型**必须** | 见 §9 里 256 GiB 那条 |

### 8.4 实测性能（Qwen2.5-0.5B，gfx1100）

```
加载 21.2 s，解码 25.86 ~ 34.78 tok/s
```
另外会看到这两条**正常**的日志，不是错误：
```
Cannot use ROCm custom paged attention kernel, falling back to Triton implementation.
cuteDSL (CUTLASS Python) not available, ll_bf16_gemm disabled
```

---

## 9. 踩过的坑 → 真因 → 修法（完整速查表）

> 用法：报错里搜关键词。**最后三条尤其重要**——它们教你怎么读日志。

### 9.1 环境 / 安装类

| 报错关键词 | 真因 | 修法 |
|---|---|---|
| `No compatible wheel found for loongarch64 at https://pypi.amd.com/vllm-rocm/...` | 设了 `VLLM_USE_PRECOMPILED=1`，它去下官方预编译 wheel，龙芯没有 | **不要设这个变量**，源码编译 |
| `externally-managed-environment` | Arch PEP 668 | 加 `--break-system-packages`（建议配 venv） |
| `Could not find a version that satisfies the requirement torch (from versions: none)` | PyPI 无 loongarch64 torch | 用 `pacman -S python-pytorch-opt-rocm`，别 pip |
| `failed-wheel-build-for-install` | pip 是**事务性**的：任何一个包 wheel 构建失败，**整条命令一个包都不会装** | 拆开单独装；别看前面一堆 `Using cached ...whl` 就以为装了 |
| `metadata-generation-failed` / `No module named 'hatch_fancy_pypi_readme'` / `ninja` | 缺构建后端 | 装 `hatch-fancy-pypi-readme scikit-build-core ninja cmake` |
| `llvm.cc` 里 `'Any' has not been declared` | 自建 LLVM 版本与 triton 源码不匹配 | §5.2 的 1 行补丁 |
| `PropertyRef` / `RegionSuccessor` / `Resource` … `2 arguments` | 用了 **LLVM 22** 编 triton 3.8 | 必须 LLVM 24（§4） |
| `LLVMNVPTX*` targets missing | LLVM 没编 NVPTX | `-DLLVM_TARGETS_TO_BUILD="AMDGPU;NVPTX"` |
| `find_package(LLD)` 失败 | 没给 lld 路径 | `-DLLD_DIR=/opt/llvm-pin/lib/cmake/lld` |
| `cuda.h: No such file or directory`（Proton） | Proton 需要 CUDA 头 | `-DTRITON_BUILD_PROTON=OFF` |
| configure 阶段卡住不动 | 在 clone googletest | `TRITON_OFFLINE_BUILD=1`（pip 会自动传 `-DTRITON_BUILD_UT=OFF`） |
| `TRITON_GSAN_CLANGXX` not found | nvidia backend 的 gsan 找 `clang++` | `ln -sf /usr/bin/clang++ /opt/llvm-pin/bin/clang++` |
| 装完 `lib/cmake/llvm` 不存在 / 只装了零星文件 | 板子时钟偏移导致 `ninja install` 跳过文件 | `rm -rf <prefix>` 再 `cmake --install <build> --prefix <prefix>` |

### 9.2 编译 / 运行类

| 报错关键词 | 真因 | 修法 |
|---|---|---|
| `unused variable 'lastLane' [-Werror=unused-variable]` | ROCm 7.2 HIP 头文件 + `-Werror` | `cmake/utils.cmake` 改 `-Wno-error=unused-variable` |
| 运行时报 `Illegal instruction detected: Invalid dpp_ctrl value ... GFX10+` | HIP 编译默认 `-O0`，LLVM 在 gfx1100 上生成非法 DPP | `GPU_FLAGS` 加 `-O2`（§6.3） |
| `LLVM ERROR: IO failure on output stream: No space left on device` | **`/tmp` tmpfs 满了**（假性磁盘满） | `truncate -s 0 /tmp/dyn-xs.log`；命令带 `TMPDIR=/var/tmp` |
| `unknown location` / `cannot import name '__version__' from 'vllm'` | 在 `/root`、`/root/vllm`… 里运行 python，命名空间包遮蔽 | `cd /tmp` 再跑 |
| `An attempt has been made to start a new process before the current process has finished its bootstrapping phase` | vLLM v1 用 spawn，脚本没有 main guard | `if __name__ == "__main__": main()`（或 `VLLM_ENABLE_V1_MULTIPROCESSING=0`） |
| `ImportError: cannot import name 'NamespaceTool' from 'openai.types.responses'` | `openai` 版本太老 | 升到 3.10.0（连带 httpx2 / httpcore2 / truststore） |
| `ModuleNotFoundError: No module named 'xgrammar' / 'tvm_ffi' / 'compressed_tensors' / 'einops' / 'blake3' / 'sentencepiece' / 'dill' / 'xxhash'` | 运行期依赖没齐 | 逐个源码装（§6.5） |
| `TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union`（栈里有 `tvm_ffi/registry.py` → `function.pxi`） | **我装的 stub 造成的**：`cuda` 被假装安装，导致 tvm_ffi 的 `try: from cuda.bindings import driver except ImportError: None` 守卫失效 | 删掉 stub，让 `import cuda` 正常失败（§10.1） |
| `AttributeError: __file__. Did you mean: '__name__'?`（栈里有 `vllm/utils/import_utils.py:68`） | **同样是 stub**：`triton_kernels` 被假装安装，`has_triton_kernels()` 返回 True 后读 `__file__` | 同上（§10.1） |
| `torch.OutOfMemoryError: Tried to allocate 256.00 GiB` / `HIPCachingAllocator … 274877906944 bytes` | 多模态模型启动时给 **ViT** 做 profiling，ROCM 上 SDPA 退化成 math 后端，物化 S×S 注意力矩阵 | 文本推理加 `language_model_only=True`（§9.3） |
| `NotImplementedError: Block pointers have been removed in favor of the tensor descriptor API` | **triton 3.8 删了 block pointer，而 vLLM 的 FLA kernel 还在用** | 见 §11（当前阻塞） |
| `RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {...}` | **这是症状，不是原因！** | 往上找 `(EngineCore pid=...)` 那段里的真错（§10.3） |
| `Cannot use ROCm custom paged attention kernel, falling back to Triton implementation.` | ROCm 无对应 paged attention kernel | **正常**，不用管 |
| `cuteDSL (CUTLASS Python) not available, ll_bf16_gemm disabled` | 没装 CUTLASS Python | **正常**，不用管 |
| `Error: could not connect to ollama server` | ollama 服务没起（`systemctl is-active ollama` → inactive） | `systemctl start ollama` |

### 9.3 关于那个 256 GiB 的 OOM（值得单独记一笔）

`Qwen3.8-27B-INT4` 是 VL 模型，vLLM 启动时**必然**给视觉编码器做一次 profiling：

```
Encoder cache will be initialized with a budget of 16384 tokens,
and profiled with 1 image items of the maximum feature size.
→ profile_run → encoder_runner.profile_encoder_cache → execute_mm_encoder → ViT → F.scaled_dot_product_attention
```

而这份 checkpoint 的 `processor_config.json` 里写着 `size = {"longest_edge": 16777216, ...}`（1677 万像素），
比 `vision_config.num_position_embeddings = 2304`（≈589824 px）**大 28 倍**；ROCm 上 Torch 没有
memory-efficient attention，SDPA 只能把完整矩阵物化出来：

```
16 heads × 65536² × 4 B = 274,877,906,944 B = 256 GiB   ← 正好等于报错里的数字
```

24 GB 的卡永远给不出 256 GiB，所以**调 `gpu_memory_utilization` 没用**。纯文本推理的解法：

```python
llm = LLM(..., language_model_only=True)     # 或命令行 --language-model-only
```

原理（已读代码确认）：`multimodal.py:495` 的 `get_limit_per_prompt()` 遇 `language_model_only` 直接返回 0
→ `encoder_budget.py:80` 的 `tower_modalities` 变空集 → `encoder_runner.py:117-125` 命中
`Skipping encoder profiling` 提前返回 → **ViT 根本不会被跑**。

---

## 10. 三个方法论教训（比具体命令更重要）

### 10.1 stub 事故：不要"假装一个包存在"

**起因**：为了让 SGLang 能 import（`quark_w4a4_mxfp4.py:86` 无守卫地 `from aiter.ops.triton... import ...`），
我装了一个全局 meta-path finder（`_optional_pkg_stubs.py` + `.pth`），让 `aiter / sgl_kernel / cuda /
triton_kernels / pynvml / xformers / deep_gemm / flashinfer …` 这些装不了的包"看起来已安装"。

**后果**：它把**调用方原本的降级分支骗成了"已安装分支"**，于是连锁炸了两处：

```
① tvm_ffi/cython/function.pxi:834
     if cuda_driver is not None and isinstance(arg, cuda_driver.CUstream):
   ← try: from cuda.bindings import driver as cuda_driver / except ImportError: cuda_driver = None
   stub 让 import 成功 → cuda_driver 是个假模块 → isinstance(arg, 假模块) → TypeError
   （表现：`from vllm import LLM` / sglang 全部在 xgrammar → tvm_ffi 处崩）

② vllm/utils/import_utils.py:68
     f"Loading module triton_kernels from {triton_kernels.__file__}."
   ← _has_module("triton_kernels") 被 stub 骗成 True → 读 __file__ → 假模块没有 → AttributeError
   （表现：EngineCore 子进程起不来，父进程只报 Engine core initialization failed）
```

**判据（请写进肌肉记忆）**：

> 只有当"这个包缺失会导致**无守卫的顶层 import 直接崩**"时，才允许 stub。
> 凡是调用方写了 `try/except ImportError` 或 `_has_module()` 的（`cuda`、`triton_kernels`、`pynvml`、
> `xformers`、`deep_gemm`、`flashinfer`、`sageattention`…）**绝对不能 stub** —— 它们的"缺失"本身就是设计要求。

**事后处置**：已全部删除，python 启动干净，vLLM 一个 stub 都不需要（实测跑通推理）。
备份在 `/root/backup-stubs-20260911/`。若将来**只**为 SGLang 需要，务必：
① 只列 aiter/sgl_kernel/sglang_kernel/flydsl/petit_kernel/wave_lang；
② 由环境变量（如 `SGLANG_STUBS=1`）开关，绝不让 vLLM 进程看到；
③ 先确认 SGLang 对这些名字有没有守卫。

### 10.2 "import 成功" ≠ "能跑"

我犯过一个具体错误：用 `from vllm import LLM` 成功来宣布"vLLM 修好了"，结果引擎照样起不来。原因是
**父进程和子进程的导入链不一样**：

```
父进程：from vllm import LLM                          → 成功
子进程 EngineCore：resolve_obj_by_qualname(worker_cls)
  → vllm/v1/worker/gpu_worker.py:59
  → model_executor/warmup/kernel_warmup.py:19
  → model_executor/warmup/deep_gemm_warmup.py:14
  → model_executor/kernels/linear/…/marlin_utils.py:14
  → model_executor/layers/fused_moe/config.py:25   has_triton_kernels()
  → 这里才炸
```

**铁律**：任何一个"能不能跑"的结论，都必须来自**真推理**：

```bash
cd /tmp && TMPDIR=/var/tmp python3 /opt/tests/vllm_smoke.py     # 加载模型 + 真的出词
```

### 10.3 怎么读 vLLM 的日志

`RuntimeError: Engine core initialization failed. See root cause above.` **永远不是根因**。
真因在这三处之一，而且往往在**更上面**：

1. `(EngineCore pid=XXXX) ERROR …` 段落里的第一条真实异常（本例：`AttributeError` / `NotImplementedError`）；
2. 没有 `EngineCore` 前缀的裸行（本例：`LLVM ERROR: IO failure … No space left on device`、
   `HIPCachingAllocator … memory allocation failed … 274877906944 bytes`）；
3. `Failed core proc(s): {}` 里**空的 `{}`** 通常意味着子进程在能注册自己之前就死了（硬崩/编译期异常）。

---

## 11. 当前唯一的阻塞项：Triton 3.8 删了 block pointer，而 vLLM 还在用

### 11.1 现象

```
(EngineCore pid=195350) triton.compiler.errors.CompilationError: at 37:14:
(EngineCore pid=195350)     else:
(EngineCore pid=195350)         p_s = tl.make_block_ptr(s + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
(EngineCore pid=195350)               ^
(EngineCore pid=195350) Block pointers have been removed in favor of the tensor descriptor API
```

### 11.2 根因链（4 步，每步都有实证）

1. **Triton 侧删除**（你编的那份）：
   `9debefe44 [BC breaking][FRONTEND] Remove block pointer support (#10833)`
   → `python/triton/language/core.py:2496-2516` 的 `make_block_ptr` / `advance` 直接 `raise NotImplementedError`；
   C++/MLIR 侧的 op 与 lowering 也一并删除（`grep -rn MakeBlockPtr include lib` 为空，**加个 Python 函数补不回来**）。
   你的 HEAD 在这个提交之后 **389 个提交**。
2. **vLLM 侧仍在使用**：全树 **323 处** `tl.make_block_ptr`：
   ```
   121  third_party/flash_linear_attention/ops/     ← 就是这里
    64  models/kimi_k3/{nvidia,amd}/ops/third_party/kda
    31  models/glm5next/{nvidia,amd}/ops/third_party/kda
     7  models/minimax_m3/... 3 models/inkling/...
   ```
   报错那行精确对应 `third_party/flash_linear_attention/ops/cumsum.py:63`。
3. **模型必须要它**：`Qwen3.8-27B-INT4`、`Qwen3.8-24B-INT4` 都是 `Qwen3_5ForConditionalGeneration`
   （混合线性注意力 / GatedDeltaNet）。其 prefill 走 vendored FLA：
   `chunk_gated_delta_rule → ops/chunk.py:37 chunk_local_cumsum → ops/cumsum.py`。
4. **没有配置项能绕**：GDN prefill 后端只有 `triton`（坏的）/ `flashinfer`（龙芯装不了）/ `cutedsl`
   （`cuteDSL not available`），代码位置 `qwen_gdn_linear_attn.py:_resolve_gdn_prefill_backend`。

### 11.3 为什么不是这些

- **不是龙芯**：x86 上用同样的 triton main + 同样的 vLLM 提交，一样炸。
- **不是 ROCm 7.2 / INT4 量化**：报错在 Triton 前端 AST 阶段，还没到 GPU。
- **不是 stub、不是 `/tmp`**：stub 已清除、推理已跑通；`/tmp` 满只需 `TMPDIR=/var/tmp`。
- **不是"换个小模型就好"**：0.5B（`Qwen2ForCausalLM`）能跑是因为**纯 full attention，根本不碰 GDN**。
  换 24B 也一样不行（同为 `qwen3_5`）。

### 11.4 三条出路（含成本）

| 路线 | 做法 | 人力 | 机时 | 评价 |
|---|---|---|---|---|
| **A. 配对 triton 版本** | 从 fork 的 `v3.5.1` tag 编 triton（ROCm 7.2 wheel index 最高就是 3.5.1，含 cp314） | 低 | **高**（3.5.1 的 LLVM pin ≠ 3.8，很可能再编一次 LLVM） | vLLM **一行不改**，323 处全恢复；只救这台机器 |
| **B. 移植上游迁移** | 把 FLA 提交 `02e27033 [Ops] Migrate kernels off tl.make_block_ptr / tl.advance for triton main compat (#1062)`（2026-07-25，88 文件，纯机械）移植到 vLLM 的 vendored 副本 | 中 | 低（迭代 5 秒级） | **真 bug fix，可提 PR** |
| **C. 换运行方式/换模型** | ①`systemctl start ollama` + `ollama run qwen3.8:27b`（模型与 80 GB blobs 都在板上，llama.cpp 不走 triton）；②或在 vLLM 里跑纯 dense 模型 | 极低 | 无 | 立刻可用，但不是"在 vLLM 里跑这个模型" |

**路线 B 的关键信息（如果要做）**：

- 上游 FLA 相关文件现在 `make_block_ptr = 0`（迁移已完成），vLLM 只是**快照太旧**。
- 这条链上只需要 8 个文件、84 处：
  `l2norm(2) → cumsum(8) → chunk_scaled_dot_kkt(4) → solve_tril(28) → wy_fast(7) → chunk_o(6) → chunk_delta_h(23) → fused_norm_gate(6)`
  （`kda.py` 的 37 处属于 Kimi/GLM，先不动。）
- 转换规则（照抄上游，别自创）：
  ```python
  # block ptr → 显式索引
  - p_s = tl.make_block_ptr(s + bos*H + i_h, (T,), (H,), (i_t*BT,), (BT,), (0,))
  + o_t = i_t * BT + tl.arange(0, BT)
  + m_t = o_t < T
  + p_s = s + bos*H + i_h + o_t * H
  # load/store → mask/other
  - b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
  + b_s = tl.load(p_s, mask=m_t, other=0.0).to(tl.float32)
  - tl.store(p_o, ..., boundary_check=(0,))
  + tl.store(p_o, ..., mask=m_t)
  # tl.advance 折进索引；order 参数丢弃（只影响向量化）
  ```
- **三个必须照抄的细节**（错一个是"不报错但结果变味"）：
  1. **索引升到 int64**（上游顺手做的：`tl.program_id(0).to(tl.int64)`、`i_t = ....to(tl.int64)`）——防长序列 int32 溢出，短 prompt 测不出来；
  2. **mask 只打 `boundary_check` 里列出的维度**，别好心给所有维都加；
  3. **不要改用 tensor descriptor**（上游这次新增 descriptor **0** 处）：很多用法最内维非单位步长（如 `strides=(H,)`，正是报错那条 `else` 分支），TMA/descriptor 表达不了；vLLM 自己也写了该限制（`fused_moe/experts/fused_batched_moe.py:97`）。
- 验证用现成测试：
  ```bash
  cd /tmp && TMPDIR=/var/tmp python3 -m pytest -x -q \
    /root/vllm/tests/kernels/mamba/ \
    /root/vllm/tests/kernels/test_fla_layernorm_guard.py \
    /root/vllm/tests/kernels/test_fused_gdn_post_conv.py \
    /root/vllm/tests/model_executor/test_qwen_triton_warmup.py
  ```
  再加一个朴素 PyTorch 参考实现做数值对照（bf16 容差 1e-3 量级）——上游该提交标了 `[skip test]`，
  **补齐验证正是最有价值的贡献**。
- 顺带值得提的 PR：给 vLLM 加一个 **triton 版本守卫**（现在只有 `TRITON3`/`TRITON_22` 这种检查，
  没有针对 block-ptr API 的），让它在启动时用一句话说清，而不是在 JIT 深处抛 `NotImplementedError`。

---

## 12. 环境自检 & 关键文件清单

### 12.1 一键自检（建议每次开工先跑）

```bash
df -h /tmp                                     # 铁律一：tmpfs 不能满
cd /tmp && python3 -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
cd /tmp && python3 -c "import triton; print(triton.__version__)"
cd /tmp && python3 -c "import vllm;  print(vllm.__version__)"
cd /tmp && python3 /opt/tests/triton_gpu_test.py    # triton 数值自检
cd /tmp && TMPDIR=/var/tmp python3 /opt/tests/vllm_smoke.py   # vLLM 端到端自检
```

### 12.2 板上的文件清单

| 路径 | 说明 |
|---|---|
| `/root/vllm` | vLLM 源码树（editable 安装），只打了 `cmake/utils.cmake` 两个补丁 |
| `/root/triton` | Triton fork（`M python/src/llvm.cc` 一行补丁） |
| `/opt/llvm-pin` | 自建 LLVM 24.0.0git（triton 用） |
| `/opt/rocm` | ROCm 7.2.0 |
| `/opt/tests/vllm_smoke.py` | vLLM 端到端冒烟测试（**判断"能不能跑"的唯一标准**） |
| `/opt/tests/triton_gpu_test.py` | triton GPU 数值自检 |
| `/opt/tests/sglang_test.py` | SGLang 离线推理测试 |
| `/opt/tests/fix_sglang_deps.sh` | SGLang 依赖补全脚本 |
| `/root/models/Qwen2.5-0.5B-Instruct` | 954 MB，**当前唯一在 vLLM 里验证通过的模型** |
| `/root/models/Qwen3.8-27B-INT4` | RedHatAI INT4，18.12 GiB checkpoint（受 §11 阻塞） |
| `/root/models/Qwen3.8-24B-INT4` | 19 GB（同样受 §11 阻塞，同为 `qwen3_5`） |
| `/root/backup-stubs-20260911/` | 已删除的 stub 文件（历史教训，见 §10.1） |
| `/root/vllm-build.log`、`/root/triton-build.log` | 构建日志 |

### 12.3 从零构建的顺序 checklist

```
0. 系统：df -h /tmp → truncate dyn-xs.log → export TMPDIR=/var/tmp ；任何时候 cd /tmp 再跑 python
1. PyTorch：pacman -S python-pytorch-opt-rocm → 验证 torch.cuda.is_available()==True
2. LLVM 24：取 llvm-project pin → cmake(mlir;lld, AMDGPU;NVPTX, RTTI, UTILS) → ninja → rm -rf + cmake --install
3. Triton：★先决定版本（ROCm 配对是 3.5.1；用 3.8 必须打 llvm.cc 补丁，且会导致 §11）
           → 按 §5.3 的命令编 → triton_gpu_test.py 数值验证
4. 构建依赖：cmake ninja hatch-fancy-pypi-readme scikit-build-core
5. vLLM 补丁：cmake/utils.cmake → -Wno-error=unused-variable + -O2
6. vLLM 编译：pip install -e . --no-build-isolation --no-deps --break-system-packages（勿设 VLLM_USE_PRECOMPILED）
7. 运行期依赖：openai≥3.10 / xgrammar / apache-tvm-ffi / compressed_tensors / einops / blake3 / sentencepiece …
8. 验证 vLLM：vllm_smoke.py 真推理（不是 import！）
9. 选模型：先看架构（§7.4）—— 纯 dense/full-attn 可直接跑；`qwen3_5`(GDN) 需先解决 §11
10. 下模型：走中转机（板上 HF/GitHub 只有几十~几百 B/s）
11. 跑推理：cd /tmp && TMPDIR=/var/tmp python3 <脚本>（脚本必须有 __main__ guard）
12. 显存调参：gpu_memory_utilization / max_model_len / enforce_eager / language_model_only
```

---

## 附录：本次会话遇到的问题的"一句话版"

1. PyPI 没有 loongarch64 的 torch/triton/vllm wheel → 全部改用发行版包 + 源码编译。
2. `VLLM_USE_PRECOMPILED=1` 会去找不存在的官方 wheel → 删掉该变量。
3. Arch PEP 668 → `--break-system-packages`；且必须 `--no-build-isolation --no-deps`。
4. ROCm HIP 头文件触发 `-Werror=unused-variable` → 改 `-Wno-error=unused-variable`。
5. HIP 默认 `-O0` 在 gfx1100 生成非法 DPP → 必须加 `-O2`（最隐蔽的一个坑）。
6. `/tmp` 是 16 GB tmpfs 且被 xserver 无限写满 → 假性"No space left on device"；`TMPDIR=/var/tmp`。
7. 在 `/root*` 目录里运行 python 会引起同名包遮蔽 → `cd /tmp`。
8. vLLM v1 用 spawn → 脚本必须 `if __name__ == "__main__":`。
9. 我装的"假包"stub 骗过了调用方的 try/except 守卫 → `tvm_ffi` TypeError、`triton_kernels` AttributeError；已全部删除。
10. 多模态模型的 ViT profiling 在 ROCm 上要 256 GiB → 文本推理用 `language_model_only=True`。
11. **当前阻塞**：triton 3.8（main）删除了 `tl.make_block_ptr`，而 vLLM 的 121 处 FLA kernel（Qwen3.5 GDN 线性注意力）仍在用 → 见 §11，两条出路由你选。
