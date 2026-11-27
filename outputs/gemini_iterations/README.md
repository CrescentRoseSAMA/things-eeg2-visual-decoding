# THINGS-EEG2 Gemini 实验迭代

任务是在 200 个未见类别上用 EEG 检索图像。正式参考 `ab_pool4` 在 10 名被试、5 个种子上取得 Top-1 12.00%、Top-5 35.73%，课程参考线为 15.6% 和 42.8%。

第十五轮的 Top-5 排序辅助损失与 QB-Norm 在 50 组正式评测中得到 **17.86% Top-1 和 43.66% Top-5**，分别高出课程参考线 2.26 和 0.86 个百分点。[项目总览与交付状态](PROJECT_SUMMARY.md) · [第十五轮正式汇总](round-15/full/summary.json) · [编码器结构图](figures/current_eeg_model.png)

三轮先导使用开发划分 2026。固定 Sub01、Sub02 的种子 0 和 Sub03、Sub04 的种子 1，每组条件共享查询、模板、训练预算和评分。先导门槛为相对 `ab_pool4` 的平均 Top-1 至少 +0.5 个百分点、至少三名被试的 Top-1 提升、平均 Top-5 持平或提高。

## 第一轮 · 四点池化下的多尺度对照

`b_pool4` 使用单尺度时间卷积和电极门控，`ab_pool4` 使用多尺度时间卷积和电极门控。两者均使用四点池化、20 个时间槽和同宽投影层。历史等参数对照中，`ab_pool4` 相对 `ab_pool8_stride4` 在 8 名被试、2 个种子上的开发集均值只高 0.0375 个百分点 Top-1、0.45 个百分点 Top-5，未达到当时的复验门槛。[历史等参数对照](../pooling_replication/summary.json)

`b_pool4` 的平均 Top-1 为 7.350%，Top-5 为 22.775%。相对 `ab_pool4` 分别变化 −0.425 和 +0.075 个百分点，四名被试中一人的 Top-1 上升，因此未过先导门槛。CSV 中的耗时来自各次运行记录，历史对照与本轮的运行条件及系统负载可能有差异。

[模型](../../eeg_model.py) · [训练](../../eeg_train.py) · [实验入口](../../eeg_gemini_iterations.py) · [协议](round-01/protocol.json) · [汇总](round-01/summary.json) · [配对数据](round-01/paired_runs.csv) · [本轮提交](https://github.com/CrescentRoseSAMA/things-eeg2-visual-decoding/commit/f43772649a481348b5138b882d179e397b1ea139)

```powershell
python eeg_gemini_iterations.py 1 --candidate b_pool4 --hypothesis "Single-scale temporal filtering with 4-point pooling and the ab_pool4 projection width"
```

## 第二轮 · 四点池化前的残差通道混合

Gemini 核对第一轮配对数据后，建议将 `Conv2d(32,32,1)`、批归一化和 ELU 组成残差分支，插在原空间读出的 ELU 与四点池化之间。`ab_pool4_pwmix` 的逐通道残差系数从零开始，因此初始输出与 `ab_pool4` 一致。新增 1,120 个参数。

平均 Top-1 为 7.425%，Top-5 为 22.625%，较 `ab_pool4` 分别变化 −0.350 和 −0.075 个百分点。仅一名被试的 Top-1 提升，未过先导门槛。四组检查点的残差系数绝对值均值依次为 0.295643、0.282681、0.253430 和 0.246075。

[模型](../../eeg_model.py) · [实验入口](../../eeg_gemini_iterations.py) · [协议](round-02/protocol.json) · [汇总](round-02/summary.json) · [配对数据](round-02/paired_runs.csv) · [本轮提交](https://github.com/CrescentRoseSAMA/things-eeg2-visual-decoding/commit/87083e0148824dc2b08361a0cbb81fc1e5707077)

```powershell
python eeg_gemini_iterations.py 2 --candidate ab_pool4_pwmix --hypothesis "Nonlinear residual channel mixing before four-point pooling, initialized as the ab_pool4 control"
```

## 第三轮 · 残差非线性投影头

Gemini 核对第二轮数据后，建议在原版 `ab_pool4` 的 768 维线性输出与 L2 归一化之间加入残差分支。`ab_pool4_resproj` 的分支由 LayerNorm、768→256 线性层、GELU、0.5 Dropout 和 256→768 线性层组成。末层权重和偏置全零初始化，初始输出与对照一致。总参数量从 493,577 增至 889,353。

平均 Top-1 为 7.375%，Top-5 为 24.850%，较 `ab_pool4` 分别变化 −0.400 和 +2.150 个百分点。四名被试的 Top-5 都提高，合计多 86 次前五命中，但只有一名被试的 Top-1 提高，合计少 16 次首位命中。这里测得的是整套残差投影头的效果，增量容量与非线性的贡献仍待分别检验。

[模型](../../eeg_model.py) · [实验入口](../../eeg_gemini_iterations.py) · [协议](round-03/protocol.json) · [汇总](round-03/summary.json) · [配对数据](round-03/paired_runs.csv)

```powershell
python eeg_gemini_iterations.py 3 --candidate ab_pool4_resproj --hypothesis "Zero-initialized LayerNorm and two-layer bottleneck residual projection before embedding normalization"
python eeg_gemini_iterations.py finalize
```

## 筛选结果

三轮候选均未达到先导门槛。预设独立复验为新类别划分 2027、Sub05–Sub10、种子 1 和 2。正式评测需独立复验平均 Top-1 至少 +1.0 个百分点、Top-5 至少 +2.0 个百分点、两个种子的平均 Top-1 均提高且至少四名被试的平均 Top-1 提高。此次停在先导，正式参考仍为 `ab_pool4`。[筛选协议](round-03/final-selection/protocol.json) · [筛选结果](round-03/final-selection/summary.json)

## 第四轮 · 可学习的对比损失温度

`ab_pool4_resproj_learn_temp` 在第三轮残差投影头上添加一个可学习标量。训练温度从 0.07 开始，其倒数最高为 100。四组沿用原开发划分及被试与种子配对，评测仍用 EEG 向量与冻结 CLIP 模板的点积。

平均 Top-1 为 6.550%，Top-5 为 20.500%。相对第三轮固定温度版分别下降 0.825 和 4.350 个百分点，相对 `ab_pool4` 分别下降 1.225 和 2.200 个百分点。四组温度在第 85–86 轮均触及 0.01 下限。本轮止于开发集。

[模型](../../eeg_model.py) · [训练](../../eeg_train.py) · [协议](round-04/protocol.json) · [汇总](round-04/summary.json) · [配对数据](round-04/paired_runs.csv)

```powershell
python eeg_gemini_iterations.py 4 --candidate ab_pool4_resproj_learn_temp --parent ab_pool4_resproj --hypothesis "Learnable InfoNCE temperature on the residual projection head, initialized at 0.07 and capped at inverse temperature 100"
```

## 第五轮 · InfoNCE 与余弦回归联合损失

`ab_pool4_resproj_cosine` 沿用第三轮的编码器与固定温度 0.07，训练损失改为 `0.75 × 对称 InfoNCE + 0.25 × (1 − 配对余弦相似度)`。四组均完成 100 轮，使用相同开发集查询与模板。

平均 Top-1 为 7.775%，Top-5 为 24.450%。相对第三轮，四名被试的 Top-1 均提高，平均变化为 +0.400 和 −0.400 个百分点。相对原版 `ab_pool4`，Top-1 持平，Top-5 提高 1.750 个百分点。本轮止于开发集。

[模型与损失](../../eeg_model.py) · [训练](../../eeg_train.py) · [协议](round-05/protocol.json) · [汇总](round-05/summary.json) · [配对数据](round-05/paired_runs.csv)

```powershell
python eeg_gemini_iterations.py 5 --candidate ab_pool4_resproj_cosine --parent ab_pool4_resproj --hypothesis "Matched-pair cosine regression at weight 0.25 added to symmetric InfoNCE with fixed 0.07 temperature"
```

## 第五轮续 · CSLS 独立复验与正式评测

固定 `k=10`，以对应被试的训练 EEG 估计模板校正项。开发复验使用类别划分 2027、Sub05–Sub10 和种子 1、2，两套模型共从头训练 24 次。候选加 CSLS 后，相对原版 Top-1 和 Top-5 分别提高 1.175 和 2.292 个百分点，两个种子的平均 Top-1 都提高，六名被试中五名的 Top-1 提高。相对同一权重仅归一化模板，CSLS 的 Top-1 提高 0.283 个百分点，达到正式评测门槛。

| 正式评分方式 | Top-1 | Top-5 |
|---|---:|---:|
| 原版 `ab_pool4` 点积 | 12.00% | 35.73% |
| 联合损失模型点积 | 12.89% | 37.00% |
| 联合损失模型，仅归一化模板 | 12.98% | 37.21% |
| 联合损失模型，余弦 CSLS | 14.70% | 39.01% |

正式阶段用全部官方训练类训练 50 份权重，全部训练结束后统一打分。较原版，CSLS 方案提高 2.70 和 3.28 个百分点，距离课程两项目标分别差 0.90 和 3.79 个百分点。归一化、点积和 CSLS 三种评分共享同一模型权重。相对仅归一化，CSLS 提升 1.72 和 1.80 个百分点，十名被试的平均 Top-1 都提高。

[复现代码](../../eeg_csls_replication.py) · [复验协议](round-05/csls_replication/protocol.json) · [复验汇总](round-05/csls_replication/summary.json) · [复验配对数据](round-05/csls_replication/paired_runs.csv) · [正式协议](round-05/csls_replication/formal/protocol.json) · [正式汇总](round-05/csls_replication/formal/summary.json) · [正式配对数据](round-05/csls_replication/formal/paired_runs.csv)

```powershell
python eeg_csls_replication.py
```

## 第六轮 · 在线试次子集平均

每个 epoch 对每张训练图等概率抽取 2、3 或 4 条 EEG 试次，无放回取样并求均值。网络保持 889,353 个参数，沿用第五轮的联合损失、温度 0.07 和 CSLS `k=10`。标准化、开发查询和 CSLS 训练库仍使用四次平均的 EEG。正式查询沿用官方平均 EEG 和 200 个固定类别模板。

| 阶段 | 配对数 | 第五轮 Top-1 / Top-5 | 第六轮 Top-1 / Top-5 | 提升（百分点） |
|---|---:|---:|---:|---:|
| 2026 先导 | 4 | 8.275% / 25.275% | 9.075% / 26.800% | +0.800 / +1.525 |
| 2027 独立复验 | 12 | 8.058% / 25.725% | 8.842% / 28.075% | +0.783 / +2.350 |
| 正式评测 | 50 | 14.70% / 39.01% | 16.64% / 40.71% | +1.94 / +1.70 |

正式评测中，十名被试的五种子平均 Top-1 均提高，八名被试的 Top-5 提高。Top-1 超过课程参考线 1.04 个百分点，Top-5 还差 2.09 个百分点。已逐条复算全部 10,000 次正式查询的排名。

[复现代码](../../eeg_trial_augmentation.py) · [先导汇总](round-06/pilot/summary.json) · [复验汇总](round-06/replication/summary.json) · [正式协议](round-06/formal/protocol.json) · [正式汇总](round-06/formal/summary.json) · [正式配对数据](round-06/formal/paired_runs.csv)

```powershell
python eeg_trial_augmentation.py pilot
python eeg_trial_augmentation.py replication
python eeg_trial_augmentation.py formal
```

## 第七轮 · 单向 CLIP 软目标

沿用第六轮的在线试次子集平均、889,353 参数编码器、联合损失和 CSLS `k=10`。仅将 EEG→图像方向的目标改为 70% 配对硬标签和 30% 批内 CLIP 图像余弦相似度分布，教师温度为 0.10。反向仍用配对硬标签。

四组开发集配对的平均 Top-1 为 8.825%，Top-5 为 26.875%，相对第六轮分别变化 −0.250 和 +0.075 个百分点。两名被试的 Top-5 提升，未达到预设的三项先导门槛。本轮止于开发集先导。

[损失](../../eeg_model.py) · [训练](../../eeg_train.py) · [实验入口](../../eeg_trial_augmentation.py) · [先导协议](round-07/pilot/protocol.json) · [先导汇总](round-07/pilot/summary.json) · [配对数据](round-07/pilot/paired_runs.csv)

```powershell
python eeg_trial_augmentation.py pilot --soft-clip
```

## 第八轮 · 跨被试预训练和目标被试微调

在 2026 开发划分的同一批训练类上用 10 名被试共同预训练 30 轮，每批取一名被试的数据。每名被试使用单位阵初始化的 `17×17` 电极适配器。四组目标被试与种子配对共享一次种子 0 的预训练，各自再微调 30 轮。保留第六轮的在线试次增强、联合损失、冻结 CLIP 和 CSLS `k=10`，CSLS 校正库仅使用目标被试的标准平均训练 EEG。

先导 Top-1 为 8.950%，Top-5 为 27.900%，较第六轮变化 −0.125 和 +1.100 个百分点。三名被试的 Top-5 提升，44 次新增前五命中中有 40 次来自 Sub03。Top-1 未达到持平门槛，本轮止于先导。共享预训练耗时 583 秒，峰值分配显存 2,084 MiB，加四组微调的训练总耗时为 772 秒。

[模型](../../eeg_model.py) · [复现代码](../../eeg_transfer_pilot.py) · [先导协议](round-08/pilot/protocol.json) · [预训练汇总](round-08/pilot/pretrain/summary.json) · [先导汇总](round-08/pilot/summary.json) · [配对数据](round-08/pilot/paired_runs.csv)

```powershell
python eeg_transfer_pilot.py
```

固定开发划分、微调种子和推进门槛，将共享预训练种子换为 1 重跑。第六轮的配对对照为 9.075% / 26.800%。

| 共享预训练种子 | Top-1 | Top-5 | 相对第六轮（百分点） | Top-5 提升被试 | 先导判定 |
|---|---:|---:|---:|---:|---|
| 0 | 8.950% | 27.900% | −0.125 / +1.100 | 3/4 | 未达标 |
| 1 | 9.625% | 28.375% | +0.550 / +1.575 | 3/4 | 达标 |

两个预训练种子下，Sub01–03 的 Top-5 都提升，Sub04 略降，Sub02 的 Top-1 都下降。种子 1 的训练耗时 786 秒。两组先导沿用同一批开发查询。

[种子 1 协议](round-08/pilot-seed-1/protocol.json) · [预训练汇总](round-08/pilot-seed-1/pretrain/summary.json) · [先导汇总](round-08/pilot-seed-1/summary.json) · [配对数据](round-08/pilot-seed-1/paired_runs.csv)

```powershell
python eeg_transfer_pilot.py --pretrain-seed 1
```

### 2027 独立复验

在与 2026 验证类别不重叠的开发划分上，从头训练两份共享权重（种子 1、2），再微调 Sub05–Sub10 的 12 组配对。第八轮为 8.650% / 27.792%，第六轮对照为 8.842% / 28.075%，差值为 −0.192 / −0.283 个百分点。种子 1、2 的平均 Top-1 分别变化 −0.417 和 +0.033 个百分点，只有 Sub05 和 Sub09 的两种子平均 Top-5 提升。未达到正式评测门槛，未使用正式测试集。两次预训练和 12 次微调合计 1,767 秒。

[复验协议](round-08/replication/protocol.json) · [复验汇总](round-08/replication/summary.json) · [逐被试逐种子数据](round-08/replication/paired_runs.csv) · [预训练种子 1](round-08/replication/pretrain/seed-1/summary.json) · [预训练种子 2](round-08/replication/pretrain/seed-2/summary.json)

```powershell
python eeg_transfer_pilot.py --replication
```

## 第九轮 · 固定 50:50 的 CSLS 分数融合

2028 划分从前两轮均未用作验证的类别中抽取 200 类。Sub05–Sub10、种子 1 和 2 的模型都在新训练划分上重训，使用相同的 1,000 条验证查询。两套融合均先各自计算 CSLS `k=10`，再等权平均分数并排序。第六轮双种子融合每名被试只生成一组排名，分别对照该被试两份单模型结果。

| 方案 | Top-1 | Top-5 | 相对第六轮单模型（百分点） | Top-5 提升被试 |
|---|---:|---:|---:|---:|
| 第六轮单模型 | 9.200% | 27.783% | — | — |
| 第六轮双种子融合 | 9.583% | 28.717% | +0.383 / +0.933 | 6/6 |
| 第六轮与第八轮融合 | 10.575% | 30.192% | +1.375 / +2.408 | 6/6 |

混合融合达到开发集门槛，且相对双种子融合提高 Top-1 0.992、Top-5 1.475 个百分点。双种子融合的 Top-5 增幅低于 1.5 个百分点门槛。课程要求每名被试分别训练和测试。含跨被试共享预训练的混合融合单列为探索结果。全程只用开发集，训练合计 3,481 秒。

[复现代码](../../eeg_score_fusion.py) · [协议](round-09/development/protocol.json) · [汇总](round-09/development/summary.json) · [配对数据](round-09/development/paired_runs.csv)

```powershell
python eeg_score_fusion.py
```

## 第十轮 · 电极残差自注意力

在第六轮模型的多尺度时间卷积前加入一层四头自注意力，以 17 个电极为 token，每个 token 保留 80 个时间点。残差系数从零开始，初始输出与第六轮一致。沿用电极门控、联合损失、在线试次子集平均、冻结 CLIP 和 CSLS `k=10`。参数量从 889,353 增至 915,434。

2026 划分四组先导的平均 Top-1 为 9.200%，Top-5 为 26.150%，较第六轮分别变化 +0.125 和 −0.650 个百分点。只有 Sub03 的 Top-5 提升（+1.4 个百分点），未达到先导门槛。四组学到的残差系数绝对值为 0.338–0.385，候选训练合计 753 秒。本轮止于开发集先导。

[模型](../../eeg_model.py) · [复现代码](../../eeg_trial_augmentation.py) · [先导协议](round-10/pilot/protocol.json) · [汇总](round-10/pilot/summary.json) · [配对数据](round-10/pilot/paired_runs.csv)

```powershell
python eeg_trial_augmentation.py pilot --electrode-attention
```

## 第十一轮 · 单试次、输入扰动与权重 EMA

四组配对先导各训练两次。两条训练轨迹都把在线平均的试次数扩为等概率 1、2、3、4，一条额外加入标准化单位下的高斯噪声（标准差 0.10）和整通道遮蔽（概率 0.05）。从第 51 轮起每轮更新一次参数 EMA（衰减 0.90），沿用该轮的 BatchNorm 缓冲区。每条轨迹都评估普通与 EMA 权重，使用标准平均 EEG、冻结 CLIP 和 CSLS `k=10`。

| 方案 | Top-1 | Top-5 | 相对第六轮（百分点） | Top-5 提升被试 |
|---|---:|---:|---:|---:|
| 第六轮对照 | 9.075% | 26.800% | — | — |
| 纳入单试次 | 9.375% | 26.975% | +0.300 / +0.175 | 1/4 |
| 纳入单试次、加输入扰动 | 9.400% | 27.350% | +0.325 / +0.550 | 2/4 |
| 纳入单试次、用 EMA | 9.200% | 27.450% | +0.125 / +0.650 | 2/4 |
| 纳入单试次、加输入扰动、用 EMA | 9.325% | 27.525% | +0.250 / +0.725 | 2/4 |

完整方案低于 Top-5 至少增加 1.0 个百分点且至少 3/4 名被试提升的门槛。相对各自的普通权重，EMA 在两条轨迹上分别提高 Top-5 0.475 和 0.175 个百分点，同时降低 Top-1 0.175 和 0.075 个百分点。八次训练合计 1,104 秒，实验止于开发集先导。

[复现代码](../../eeg_trial_regularization.py) · [先导协议](round-11/pilot/protocol.json) · [汇总](round-11/pilot/summary.json) · [配对数据](round-11/pilot/paired_runs.csv)

```powershell
python eeg_trial_regularization.py
```

## 第十二轮 · 展平时空输入的 EEGProject

将 `17×80` EEG 展平，经 `Linear(1360,768)`、由 GELU、线性层和 0.3 Dropout 构成的残差分支、LayerNorm 和 L2 归一化输出。保留第六轮的 Adam、100 轮、在线试次子集平均、联合损失和 CSLS `k=10`。模型有 1,637,376 个参数。

2026 四组先导的平均 Top-1 为 6.475%，Top-5 为 20.925%，相对第六轮变化 −2.600 和 −5.875 个百分点。四名被试的 Top-5 都下降，未达到先导门槛。训练合计 279 秒。

将两份模型的 CSLS 分数按 50:50 平均后，得到 9.050% / 26.800%，相对第六轮变化 −0.025 / 0.000 个百分点。纯 MLP 和 CNN 各有 262 和 497 次只有自己命中的前五查询。该诊断沿用同一开发集，本轮没有训练 CNN＋MLP 双分支。

[模型](../../eeg_model.py) · [复现代码](../../eeg_trial_augmentation.py) · [先导协议](round-12/pilot/protocol.json) · [汇总](round-12/pilot/summary.json) · [配对数据](round-12/pilot/paired_runs.csv)

```powershell
python eeg_trial_augmentation.py pilot --eegproject
```

## 第十三轮 · 固定权重的 QB-Norm 与 CSLS 对照

严格重载第六轮的四份先导权重，用每名被试自己的 14,540 条标准四次平均训练 EEG 建校正库。冻结 CLIP 模板、余弦分数、查询和权重在两种评分方式间共享。候选使用动态倒置 softmax（QB-Norm `β=20、k=1`），对照用原版 CSLS `k=10`。本轮没有新增训练。

QB-Norm 的 Top-1 为 9.150%，Top-5 为 27.175%，较第六轮分别提高 0.075 和 0.375 个百分点。四名被试的 Top-5 都提高，合计新增 98 次前五命中、失去 83 次，净增 15 次。平均 Top-5 涨幅低于 1.0 个百分点的门槛。

[复现代码](../../eeg_qb_csls_pilot.py) · [先导协议](round-13/pilot/protocol.json) · [汇总](round-13/pilot/summary.json) · [配对数据](round-13/pilot/paired_runs.csv)

```powershell
python eeg_qb_csls_pilot.py
```

### 第十三轮补充 · 50 组正式测试集评分

先导未达到原推进门槛。按后续要求，沿用先导锁定的 `β=20、k=1`，对第六轮已有的 10 名被试 × 5 个种子权重做全量补充评分。每组从该被试的 16,540 条官方训练 EEG 平均样本建立校正库，用 200 条正式测试查询和既有 CLIP 模板评分。两种评分共用同一份权重，新增训练数为 0。

QB-Norm 得到 **17.44% / 42.43%**，同权重 CSLS 为 **16.64% / 40.71%**，提升 **0.80 / 1.72 个百分点**。10 名被试的五种子平均 Top-5 均提高。QB-Norm 的五种子均值之间的标准差为 0.385 / 0.084 个百分点，距课程 Top-5 参考线还差 0.37 个百分点。已独立复算全部 10,000 条查询的分数和排名。

[全量协议](round-13/full/protocol.json) · [全量汇总](round-13/full/summary.json) · [逐被试逐种子数据](round-13/full/paired_runs.csv)

```powershell
python eeg_qb_csls_pilot.py --full
```

## 第十四轮 · 降低训练期高斯噪声

在第十一轮的单试次采样、通道遮蔽和 EMA 方案上，将训练期高斯噪声的标准差由 0.10 改为 0.05。2026 开发划分的四组配对沿用相同的训练种子和查询，三组方案统一使用 QB-Norm `β=20、k=1`。新增四次训练。

| 方案 | Top-1 | Top-5 |
|---|---:|---:|
| 第六轮 + QB-Norm | 9.150% | 27.175% |
| 第十一轮 EMA，噪声 0.10 + QB-Norm | 9.750% | 27.850% |
| 噪声 0.05 + EMA + QB-Norm | 9.750% | 27.875% |

降低噪声后，4,000 条查询净增 1 次前五命中。相对第六轮加 QB-Norm，Top-5 提高 0.700 个百分点，只有两名被试提升。Sub01 和 Sub02 分别下降 1.1 和 0.6 个百分点。本轮未达到开发集先导门槛。

[复现代码](../../eeg_qb_noise_pilot.py) · [先导协议](round-14/pilot/protocol.json) · [汇总](round-14/pilot/summary.json) · [配对数据](round-14/pilot/paired_runs.csv)

```powershell
python eeg_qb_noise_pilot.py
```

## 第十五轮 · Top-5 排序辅助损失

沿用第六轮骨干和训练设置，在原联合损失上加权重 0.10 的排序损失。训练时每条 EEG 使用 1 个正确类和 199 个训练集错误类，正确类模板排除当前配对图像。只有第 5 高的错误类分数超过正确类分数时才产生惩罚。评测使用固定 QB-Norm `β=20、k=1`。

2026 开发集四组配对中，候选得到 9.525% Top-1 和 27.325% Top-5，相对第六轮加 QB-Norm 分别提高 0.375 和 0.150 个百分点。四名被试的 Top-1 均提高，两名被试的 Top-5 提高。平均 Top-5 增幅和提升被试数均低于先导门槛。

[复现代码](../../eeg_top5_qb_pilot.py) · [先导协议](round-15/pilot/protocol.json) · [汇总](round-15/pilot/summary.json) · [配对数据](round-15/pilot/paired_runs.csv)

```powershell
python eeg_top5_qb_pilot.py
```

### 第十五轮补充 · 50 组正式测试

应用户要求，沿用先导锁定的排序损失和 QB-Norm 设置，按被试独立从头训练 10 名被试 × 5 个种子的 50 份权重，每组覆盖全部 1,654 个官方训练类别，训练 100 轮。训练中断后从已保存的断点续跑。全部训练完成后才统一对每组的 200 条正式测试查询评分。第十三轮的 QB-Norm 结果作为逐项配对对照。

| 方案 | Top-1 | Top-5 |
|---|---:|---:|
| 第十五轮排序损失 + QB-Norm | **17.86%** | **43.66%** |
| 第十三轮对照 | 17.44% | 42.43% |
| 课程参考线 | 15.60% | 42.80% |

相对第十三轮，Top-1 和 Top-5 分别提高 0.42 和 1.23 个百分点。9/10 名被试的五种子平均 Top-5 提高。五个种子均值的样本标准差分别为 0.222 和 0.605 个百分点。两项准确率分别高于课程参考线 2.26 和 0.86 个百分点。已独立复算两组各 10,000 条查询的 QB-Norm 分数、排名和命中数。

[全量协议](round-15/full/protocol.json) · [全量汇总](round-15/full/summary.json) · [逐被试逐种子配对数据](round-15/full/paired_runs.csv)

```powershell
python eeg_top5_qb_pilot.py --full
```
