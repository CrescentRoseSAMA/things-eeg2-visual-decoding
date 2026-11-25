# THINGS-EEG2 Gemini 实验迭代

任务是在 200 个未见类别上用 EEG 检索图像。正式参考 `ab_pool4` 在 10 名被试、5 个种子上取得 Top-1 12.00%、Top-5 35.73%，课程参考线为 15.6% 和 42.8%。

三轮先导使用开发划分 2026。固定 Sub01、Sub02 的种子 0 和 Sub03、Sub04 的种子 1，每组条件共享查询、模板、训练预算和评分。先导门槛为相对 `ab_pool4` 的平均 Top-1 至少 +0.5 个百分点、至少三名被试的 Top-1 提升、平均 Top-5 持平或提高。

## 第一轮 · 四点池化下的多尺度对照

`b_pool4` 使用单尺度时间卷积和电极门控，`ab_pool4` 使用多尺度时间卷积和电极门控。两者均使用四点池化、20 个时间槽和同宽投影层。历史等参数对照中，`ab_pool4` 相对 `ab_pool8_stride4` 在 8 名被试、2 个种子上的开发集均值只高 0.0375 个百分点 Top-1、0.45 个百分点 Top-5，未达到当时的复验门槛。[历史等参数对照](../pooling_replication/summary.json)

`b_pool4` 的平均 Top-1 为 7.350%，Top-5 为 22.775%。相对 `ab_pool4` 分别变化 −0.425 和 +0.075 个百分点，四名被试中一人的 Top-1 上升，因此未过先导门槛。CSV 中的耗时来自各次运行记录，历史对照与本轮的运行条件及系统负载可能有差异。

[模型](../../eeg_model.py) · [训练](../../eeg_train.py) · [实验入口](../../eeg_gemini_iterations.py) · [协议](round-01/protocol.json) · [汇总](round-01/summary.json) · [配对数据](round-01/paired_runs.csv)

```powershell
python eeg_gemini_iterations.py 1 --candidate b_pool4 --hypothesis "Single-scale temporal filtering with 4-point pooling and the ab_pool4 projection width"
```
