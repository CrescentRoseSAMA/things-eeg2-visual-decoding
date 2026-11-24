# THINGS-EEG2 视觉解码实验

用被试观看图像时的 EEG 信号，在 200 个候选图像类别中做零样本检索。实验使用 17 通道 EEG、刺激后的 80 个采样点和冻结的 CLIP 图像特征。仓库提供代码、实验协议、汇总结果及课程资料。

## 实验结果

四点时间池化模型 `ab_pool4` 在 10 名被试、每人 5 个随机种子的正式评测中，平均 Top-1 为 12.00%，Top-5 为 35.73%。逐次结果见[正式评测](outputs/pooling_formal/summary.json)。

最近的开发集实验为每条 EEG 增加 16 个异类图像负样本。两名先导被试的平均 Top-1 分别为原版 7.40%、随机负样本 7.35%、视觉近邻负样本 7.30%。每组使用相同验证查询；这轮试验止于开发集。[实验协议](outputs/visual_negatives/protocol.json) · [逐被试结果](outputs/visual_negatives/pilot/arm_counts.csv)

## 复现

运行环境和数据来源见[实验运行说明](README_EXPERIMENT.md)。原始 EEG、CLIP 特征、模型检查点和完整材料包在本地生成或按说明获取。下列命令假设已经准备好原始输入，并已安装实验依赖。

```powershell
python -m unittest discover -s tests
python eeg_data.py
python eeg_train.py smoke
python eeg_train.py verify-smoke
python eeg_train.py develop
python eeg_pooling_dev.py
python eeg_negative_experiment.py
```

## 课程资料

- [课程要求原图](2026数据挖掘技术期末专题大作业/2026数据挖掘技术期末专题大作业/2026数据挖掘技术期末专题大作业.png) · [课程作业模板](2026数据挖掘技术期末专题大作业/2026数据挖掘技术期末专题大作业/数据挖掘技术期末专题大作业模板.doc)
- [课堂录音转写全文](analysis/_evidence/checkpoints/job_40c402ddfcaaf795b9fc.md) · [原始转写 TXT 和 JSON](analysis/_evidence/raw) · [补充转写](analysis/_evidence/supplements)
- [转录校订记录](analysis/03_转录校订记录.md) · [课堂总结与作业要求对照](analysis/04_课堂总结与作业关联报告.md)

转写来自课堂录屏音轨的自动识别。遇到术语、数字或提交要求的歧义，可对照课程要求原图和校订记录。
