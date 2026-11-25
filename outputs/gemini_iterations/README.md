# THINGS-EEG2 Gemini 实验迭代

任务是在 200 个未见类别上用 EEG 检索图像。正式参考 `ab_pool4` 在 10 名被试、5 个种子上取得 Top-1 12.00%、Top-5 35.73%，课程参考线为 15.6% 和 42.8%。

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
