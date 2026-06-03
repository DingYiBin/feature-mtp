# Feature for multi tokens Prediction
## Background
Decode only 架构的 LLM（如 GPT、LLaMA 系列），推理时采用自回归（autoregressive）方式：每步只输入一个 token，模型基于已生成的全部 token 预测下一个 token。具体做法是：

1. **逐 token 生成**：将 prompt 输入模型，得到第一个输出 token；然后将该 token 拼接到输入末尾，再预测下一个 token，如此反复。
2. **KV Cache**：为了避免每步重复计算前面 token 的 Key 和 Value，推理时会维护一个 KV 缓存。每步只计算当前 token 的 K/V，追加到缓存中，后续注意力层直接从缓存读取完整序列的 K/V。
3. **单步推理延迟**：每步生成一个 token，因此总生成时间 = 单步推理延迟 × 输出 token 数。单步推理的时延主要来自注意力计算和 FFN 计算。

这种逐 token 串行生成的方式在输出长序列时效率较低，因此衍生出 speculative decoding 等加速方法。
Medusa、EAGLE 和 MTP 同属多 token 预测推理加速方案，但它们的核心设计理念和适用场景截然不同。


### medusa/eagle/MTP

多 token 预测推理加速方案的核心思想是 "**猜 — 并行验证**"（Draft & Verify）：由一个快速草稿模块一次性生成多个未来候选 token，再由目标大模型在单次前向传播中对这些候选 token 进行并行验证，从而将原本需要 K 步自回归生成的过程压缩为 1 步，显著降低每 token 的平均推理延迟。然而，不同方案在 "如何猜" 和 "如何验证" 这两个关键环节上采取了迥异的设计思路，以下逐一展开。

**Medusa：轻量多头并行解码**

Medusa 的核心设计是在冻结的主干模型之上附加多个轻量级解码头（Medusa Heads），每个头负责预测一个特定偏移位置的未来 token。

在训练阶段，Medusa 提供了两个层级：**Medusa-1** 保持骨干 LLM 完全冻结，仅微调解码头，实现无损加速；**Medusa-2** 则允许骨干模型与解码头联合微调，预测精度更高，但需要专门的训练方案以维护模型原有能力。在推理阶段，主干模型只需一次前向传播，多个 Medusa Head 基于同一 hidden state 并行输出 t+1、t+2……t+n 位置的候选 token，无需 token 间的串行依赖。

**EAGLE：特征重用的轻量自回归草稿模型**

EAGLE 直接在目标模型的 hidden state 之上，接一个极轻量的 Transformer 层（通常仅 1 层，参数量为目标模型的 1%~5%）作为草稿生成器，利用目标模型已经计算好的特征进行推测。

EAGLE 采用**自回归式起草**（Autoregressive Drafting）：草稿模型依次生成 K 个 token，每次将上一步的预测 token 与对应的 hidden state 一并输入下一轮起草。这意味着 EAGLE 的草稿生成过程本身有 token 依赖，需要在草稿模型内部执行 K 次前向传播。但正因如此，EAGLE 生成的候选 token 序列天然保持了 token 间的依赖关系，验证阶段的接受率显著更高。最终，目标模型在一次 Batch 前向中验证整棵候选树（树结构通过 Top-K 分叉构建，路径概率采用 n-gram 连乘打分），接受最长的正确前缀。

**MTP：训练阶段的未来感知多 token 预测**

与前两者不同，MTP（Multi-Token Prediction）首先是一种**训练目标设计**，而非单纯的推理加速插件。它由 DeepSeek-V3 技术报告系统化引入并推广，通过在预训练阶段让模型在每个位置上同时预测多个未来 token（t+1、t+2……t+n），使 hidden state 必须编码更长远的信息，从而提升模型的数据效率和表征质量。

在架构层面，MTP 在共享的主干 Transformer 之上附加多个预测头（或 MTP Module），每个对应一个未来偏移位置。训练时采用链式 Teacher Forcing：第 k 个头的输入依赖于前 k-1 个头的**真实 token（Ground Truth）** ，形成训练阶段的序列依赖，损失函数为主损失与各层 MTP 损失的加权求和：total_loss = main_loss + (avg_mtp_loss × scaling_factor)。推理阶段可以将 MTP 模块**复用为内置草稿路径**，与主模型构成自推测解码（Self-Speculative Decoding），无需外部草稿模型即可实现推理加速。

MTP 的核心优势在于**训练即获得双重收益**：一方面通过更密集的训练信号提升模型本身的预测质量（已在 DeepSeek-V3、Qwen3-Next、Nemotron 3 Super 等模型中验证），另一方面可复用于推理加速，无需额外部署草稿模型。训练开销极小。

## Feature for multi tokens prediction
但是，medusa/eagle/mtp 这些方案都是基于预测下一个 token 的模型。尽管 mtp 的接受率远高于 medusa、eagle，在特定的case（coding）中使用3层MTP，可以达到 3.6+ 的平均接收长度。但是这并没有解决笔者的疑问：

**有没有可能同时预测多个token，确定性地预测多个token？**

或者这么问：

**大模型能否预测未来m个token的feature，然后再通过一个结构，将其解码为m个token？**
### mtp

重新审视 mtp 的架构：
![mtp](figure/mtp.png)

对于一次 decode，输入的是 token $t_p$, 经过 embedding 得到 $e_p$，经过 main model 得到 $h_p$, $h_p$ 输入给 lm head，得到 $prob_p$, sample 得到 token $t_{p+1}$, 然后这个 token 得到 $e_{p+1}$ 和$h_p$ 作为 $mtp_1$ 的输入，得到 $h_p^1$，可以得到 $t_{p+2}$。

换而言之，我们可以把一次 decode 视为一个 per postion，都有不同权重的 RNN：

![rnn-mtp](figure/rnn-mtp.png)

很容易注意到，如果把一次decode视为一个rnn的执行，main model的权重太大了，远大于后续用于生成其他 token 的权重。换句话说，这个流程下，预测的 p+1 的 token 应该比 p+2 的 token 更准确，这也是我们进行 speculative decode 时，总是以 main model 的概率为准的核心原因。

我们从一个实验中也可以看到这一点，实验设置如下
- 数据集：以一个 RL 数据集 ([a-m-team/AM-DeepSeek-Distilled-40M](https://huggingface.co/datasets/a-m-team/AM-DeepSeek-Distilled-40M)) 为主构造的训练集（合计约100B token+）上。
- 模型结构：Qwen3-4B 的结构为主干，lm head 和 embedding 权重独立，7层 mtp
- 训练设置：mtp scale设置为 $0.1$ (TODO 7)

经过2048步后，可以看到：

lm loss: 1.075361E+00 | mtp_1 loss: 1.464631E+00 | mtp_2 loss: 1.522981E+00 | mtp_3 loss: 1.534986E+00 | mtp_4 loss: 1.549159E+00 | mtp_5 loss: 1.557763E+00 | mtp_6 loss: 1.550024E+00 | mtp_7 loss: 1.548013E+00

经过 16384 步后：

lm loss: 6.860194E-01 | mtp_1 loss: 9.684212E-01 | mtp_2 loss: 1.020181E+00 | mtp_3 loss: 1.040076E+00 | mtp_4 loss: 1.055723E+00 | mtp_5 loss: 1.066478E+00 | mtp_6 loss: 1.069652E+00 | mtp_7 loss: 1.073924E+00 

可以注意到，mtp 的几个 loss 随 position 上升，但是差较小，远小于 lm loss 和 mtp1 loss 的差。这是否意味着，如果预测的下一个 token 的结构和后续的 mtp 层保持一致，就可以得到一个逐 position 的 loss 差异更小的解码模块：

![fmtp0](figure/fmtp0.png)

很容易注意到 mtp 和 main model 共享 embedding 层，这也就意味着 postion p 上的 token 对应相同的 embedding 输入给整个模型两次，所以，是否 mtp 层使用独立的 embeding更合适？

![fmtp1](figure/fmtp1.png)

总之，我选择了 mtp 层共用一个和 main model 不同的 embedding。


但是，这个比较可能并不合理：
1. fmtp 的 lm loss 是 8 个 position 上的 loss 的均值。
2. original mtp 的 loss，lm loss(对应 fmtp 的 pos 0 的 loss)，系数为1，mtp scale 默认为 0.1，因此每层的系数为 $1/70$

因此，我们或许还需要两个实验：
1. fmtp 的 lm loss 为 8 个 position 的 loss 和。
2. original mtp 设置 mtp scale 为 7

当然，original mtp 设置 mtp scale 为 7 后，lm loss 和 mtp 0 loss 的差异仍大于 fmtp 的差值，也可以证明 fmtp 可能确实可以支持得到一个能同时预测接下来 m 个 token 的模型。
## something useless
