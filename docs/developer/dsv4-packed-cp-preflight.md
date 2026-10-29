# DSv4 PR02 packed CP 训练预检（2026-10-29）

本次范围是 PR02 继承最新 PR01 后的 packed CP 训练流程。更新后的代码已在本机
8 卡完成 CP2、CP4 和 CP2+mHC 的短程训练，可作为业务流程预跑基线。

## 代码基线

- PR01：`94a5b83e54`，包含后端兼容保护、mHC expert overlap 修复及格式修订。
- 原 PR02：`a74db9865f`，原共同基线为 `a7ca1417df`。
- 更新分支：`pr/dsv4-02-packed-cp-refresh`。
- 工作树：`.worktrees/pr-dsv4-02-packed-cp-refresh`。

PR02 的 5 个提交已经重放到 PR01。唯一冲突在
`fine_grained_callables.py`：保留 PR01 的 mHC 执行逻辑；它已经从
`node.chunk_state.padding_mask` 读取并传递路由 padding mask。
原 PR01/02/03 和训练工作树保持原状。没有合入 PR03 或训练树的优化提交。

本次额外修复：

1. Packed 预训练启用 `dataloader_inter_document_masking` 时，在中间
   PP/VPP stage 也构建数据集，满足 `get_batch()` 对文档边界的需求。
2. H100/SM90 的 cuDNN sparse indexer score/backward 要求槽位按 128 对齐。
   top-k=16 原来产生零长度寄存器张量；现在仅对损失输入补 `-1` 无效槽位，
   保持 attention 选中的 key 不变。
3. 数据和 optimizer 回归用例适配 8 ranks，新增 PP4×CP2 中间 stage 元数据回归。

## 运行入口

在新工作树根目录、已经安装 PyTorch/TE/FlashMLA/cuDNN DSA 的环境中运行：

```bash
export PYTHONPATH=.
export NCCL_NVLS_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  tools/dsv4_packed_cp_smoke.py --cp-size 2
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  tools/dsv4_packed_cp_smoke.py --cp-size 4
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  tools/dsv4_packed_cp_smoke.py --cp-size 2 --enable-hyper-connections
```

脚本使用真实三层 GPT，压缩比例 `[0, 4, 128]`，TP1、静态 contiguous CP、
BF16、cuDNN/FlashMLA 和 sparse indexer loss。三个 batch 分别包含不同数量和边界的
文档，覆盖空文档和内部 padding。每步两个 microbatch，检查梯度和更新的有限性、
CP 副本参数抽查，并进行带 momentum 的 SGD 状态保存、恢复和下一步重放。
恢复后的参数要求逐位一致；继续训练一步的 loss 使用 `rtol=atol=1e-5`，
参数使用 PyTorch 的标准 dtype 容差。cuDNN packed backward 不保证逐位确定性，
脚本会另外报告下一步参数是否逐位相同及最大绝对差。

每个 CP 组独立运行相同合成小模型。这个入口验证普通调度和本地 `state_dict`
恢复，不代表业务数据、分布式 optimizer/checkpoint 或业务规模的验收。

业务训练配置应保持以下约定：

| 配置 | 预检范围 |
| --- | --- |
| `experimental_attention_variant` | `dsv4_hybrid` |
| `multi_latent_attention` | `True` |
| `tensor_model_parallel_size` | `1` |
| `context_parallel_size` | 固定为 `2` 或 `4` |
| `attention_cp_layout` | `contiguous` |
| `dsa_kernel_backend` | 显式 `cudnn` |
| `dsa_indexer_use_sparse_loss` | H100 上设为 `True` |
| `calculate_per_token_loss` | `True`，按真实 token 数归一化，适配各 CP rank 不同的 padding 数量 |
| packed 预训练入口 | `--dataloader-inter-document-masking` |
| packed SFT 入口 | `--sft`，沿用 SFT 文档元数据 |
| `PackedSeqParams` | `qkv_format='thd'`；保留真实和物理 padded 两套累计长度 |

从普通调度、MTP 关闭的配置开始对接业务。每个 CP rank 的 token 数需满足
attention 左边界交换窗口的长度要求。动态 CP 和确定性 packed backward 不在此实现范围内。

## 验证记录

服务器：8×H100 80GB；PyTorch `2.9.0+cu130`，Transformer Engine `2.14.0`。
实验由 GPT-6.1-sol、低推理子 agent 执行；每次使用 8 ranks。

最终仓库版 `tools/dsv4_packed_cp_smoke.py` 的结果：

| 模式 | 8 ranks 结果 | 三步变包训练 | checkpoint 恢复及下一步重放 |
| --- | --- | --- | --- |
| TP1 / CP2 | 全部通过，launcher 退出 0 | 通过 | 通过 |
| TP1 / CP4 | 全部通过，launcher 退出 0 | 通过 | 通过 |
| TP1 / CP2 / mHC | 全部通过，launcher 退出 0 | 通过 | 通过 |

三种模式均未触发编译缓存耗尽。恢复后的参数逐位一致，下一步的 loss 和参数通过
上述容差检查；跨所有 rank 的最大重放参数绝对差为 `6.103515625e-5`。

其他回归结果：

- packed attention 首轮：每 rank 20 passed、2 skipped、4 failed。
  两个 skip 是需要 SM100 的 dense indexer loss 场景。
- SM90 槽位修复后，4 个失败场景的 packed CP 对 CP1 比较全部通过，
  最低 cosine 为 `0.999989718`；CP4 的两个完整用例通过。
- CP2 的两个完整用例仍在附带 SBHD fused/native 对照上失败，属于非 packed
  indexer 梯度比较：最低 cosine `0.998839240`，阈值 `0.999`；失败张量的
  最大绝对差 `6.92e-7`。没有放宽阈值，不能把整份 suite 宣称为全绿。
- layout、packed batch、PP4×CP2 元数据、真实 PP2 GPT 前后向以及 router padding
  回归：全部 8 ranks 各 41 passed。

完整实验日志位于 `/tmp/dsv4-pr02-gpu-20261029/`，包括每个 rank 的 stdout/stderr，
`SUMMARY.md` 中的完整运行命令、`final-smoke-results.json` 中的逐 rank 结果，
以及 `sparse-parity-failures.json` 中逐张量的数值差异。
