"""Generate numbered source tables, figures, and the Word laboratory report."""
import csv
import shutil
from datetime import date
from pathlib import Path
from statistics import mean

import numpy as np

from eeg_evaluate import CONFIGS, SUBJECTS, write_csv


NAMES = {"baseline": "基线", "a": "仅 A", "b": "仅 B", "ab": "A+B"}
PAPERS = Path(__file__).with_name("experiment_literature.csv")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def figures(args, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample = Path(__file__).resolve().parent / "reports/figures/sub01_waveform_input.npz"
    if sample.is_file():
        with np.load(sample) as source:
            single, times = source["eeg"], source["times"]
            channel_names = source["ch_names"].tolist()
    else:
        example = args.data / "Preprocessed_data/sub-01/preprocessed_eeg_training.npy"
        dataset = np.load(example, allow_pickle=True).item()
        single = np.array(dataset["preprocessed_eeg_data"][0], dtype=np.float64)
        times = np.asarray(dataset["times"], dtype=float)
        channel_names = list(dataset["ch_names"])
        del dataset
        sample.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sample, eeg=single, times=times, ch_names=np.asarray(channel_names))
    times = np.round(times, 6)
    averaged = single.mean(axis=0)
    oz = channel_names.index("Oz")
    path = out / "图1_单次与重复平均"
    wave = [{"time_s": t, "repeat1_Oz": x, "mean4_Oz": y}
            for t, x, y in zip(times, single[0, oz], averaged[oz])]
    write_csv(path.with_suffix(".csv"), wave)
    fig, ax = plt.subplots(figsize=(7.1, 2.7), constrained_layout=True)
    ax.plot(times, single[0, oz], color="#8f9ca8", linewidth=1, label="Single presentation")
    ax.plot(times, averaged[oz], color="#244c73", linewidth=1.8, label="Mean of 4 presentations")
    ax.axvline(0, linewidth=0.8, color="black", linestyle=":")
    ax.set(xlabel="Time relative to stimulus onset (s)", ylabel="Signal (file units)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path.with_suffix(".png"), dpi=170)
    plt.close(fig)

    path = out / "图2_电极时间响应"
    write_csv(path.with_suffix(".csv"), [dict(time_s=t, **dict(zip(channel_names, col)))
                                         for t, col in zip(times, averaged.T)])
    fig, ax = plt.subplots(figsize=(7.1, 3.4), constrained_layout=True)
    bound = max(abs(averaged.min()), abs(averaged.max()), 1e-12)
    im = ax.imshow(averaged, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound,
                   extent=(times[0] - .005, times[-1] + .005, len(channel_names) - .5, -.5))
    ax.set_yticks(range(len(channel_names)), channel_names, fontsize=7)
    ax.set(xlabel="Time relative to stimulus onset (s)", ylabel="Electrode")
    ax.axvline(0, linewidth=.7, color="black", linestyle=":")
    fig.colorbar(im, ax=ax, label="Signal (file units)")
    fig.savefig(path.with_suffix(".png"), dpi=170)
    plt.close(fig)

    dev = read_csv(args.dev / "ab/sub-01/seed-0/train.csv")
    path = out / "图3_开发训练与验证"
    write_csv(path.with_suffix(".csv"), dev)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), constrained_layout=True)
    epochs = [int(r["epoch"]) for r in dev]
    axes[0].plot(epochs, [float(r["loss"]) for r in dev], color="#244c73")
    axes[0].set(xlabel="Epoch", ylabel="Training loss")
    for key, label, color in (("val_top1", "Top-1", "#244c73"), ("val_top5", "Top-5", "#a85826")):
        axes[1].plot(epochs, [100 * float(r[key]) for r in dev], label=label, color=color)
    axes[1].set(xlabel="Epoch", ylabel="Validation accuracy (%)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(path.with_suffix(".png"), dpi=170)
    plt.close(fig)

    path = out / "图4_EEG与候选模板相似度"
    scores = np.load(args.runs / "ab/sub-01/seed-0/scores.npy", allow_pickle=False)
    np.save(path.with_suffix(".npy"), scores)
    fig, ax = plt.subplots(figsize=(5.5, 4.3), constrained_layout=True)
    im = ax.imshow(scores, cmap="viridis", aspect="equal", interpolation="nearest")
    ax.set(xlabel="Template index", ylabel="EEG query index")
    fig.colorbar(im, ax=ax, label="Matching score")
    fig.savefig(path.with_suffix(".png"), dpi=170)
    plt.close(fig)


def build_report(args, rows, overall, subjects, deltas):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    out = args.out / "图表"
    out.mkdir(parents=True, exist_ok=True)
    figures(args, out)
    papers = read_csv(PAPERS)
    doc = Document()
    page = doc.sections[0]
    page.page_width, page.page_height = Cm(21), Cm(29.7)
    page.top_margin, page.bottom_margin = Cm(2.3), Cm(2.3)
    page.left_margin, page.right_margin = Cm(2.5), Cm(2.5)
    for style in ("Normal", "Title", "Heading 1", "Heading 2", "Caption"):
        spec = doc.styles[style]
        spec.font.name = "Times New Roman"
        spec.font.size = Pt(10.5 if style in ("Normal", "Caption") else (20 if style == "Title" else 14))
        spec.font.color.rgb = RGBColor(0, 0, 0)
        spec._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
        spec.paragraph_format.line_spacing = 1.5
        spec.paragraph_format.space_after = Pt(4)
    p = doc.add_paragraph
    head = doc.add_heading

    def table(number, title, headers, values):
        tabpath = out / f"表{number}_{title}.csv"
        write_csv(tabpath, [dict(zip(headers, v)) for v in values], headers)
        cap = p(f"表 {number}　{title}", style="Caption")
        cap.paragraph_format.keep_with_next = True
        tab = doc.add_table(rows=1, cols=len(headers))
        tab.style = "Table Grid"
        tab.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        for i, text in enumerate(headers):
            tab.rows[0].cells[i].text = text
        for fields in values:
            row = tab.add_row()
            row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
            for cell, text in zip(row.cells, fields):
                cell.text = str(text)
        for row in tab.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.paragraph_format.space_after = Pt(0)

    def pic(stem, caption, width=15.5):
        body = p()
        body.alignment = WD_ALIGN_PARAGRAPH.CENTER
        body.paragraph_format.keep_with_next = True
        body.add_run().add_picture(str(out / (stem + ".png")), width=Cm(width))
        cap = p(caption, style="Caption")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER

    title = p("数据挖掘技术期末专题大作业", style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(72)
    subtitle = p("17 通道 EEG 的零样本视觉解码")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(40)
    for label, value in (("学院", args.college), ("班级", args.class_name), ("姓名", args.name),
                         ("学号", args.student_id), ("日期", date.today().isoformat())):
        line = p(f"{label}　{value or '________________'}")
        line.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break()
    head("任务与研究问题", 1)
    full = next(r for r in overall if r["config"] == "ab")
    attained = full["top1_pct"] > 15.6 + 1e-10 and full["top5_pct"] > 42.8 + 1e-10
    p(f"本实验用同一名被试的训练数据学习 EEG 到冻结视觉空间的映射，在 200 个未见类别中识别刺激。"
      f"主模型跨 10 名被试、5 个随机种子的 Top-1 为 {full['top1_pct']:.2f}%，Top-5 为 {full['top5_pct']:.2f}%。"
      f"相比 NICE-GA，分别相差 {full['top1_vs_reference_pp']:+.2f} 和 {full['top5_vs_reference_pp']:+.2f} 个百分点，"
      f"{'两项均超过课程参考值' if attained else '未同时超过两项课程参考值'}。")
    p("用同一组图像模板比较四个模型，分别检验不同时间尺度的互补性，以及按样本调整电极权重的贡献。"
      "其他文献的任务、候选集和评价指标各异，表 1 列出每项工作在本实验中支持的具体选择。")
    table(1, "文献对照", ("研究", "任务与协议", "指标", "设计依据"),
          [(r["short"], r["task_protocol"], r["metric"], r["experiment_use"]) for r in papers])
    doc.add_page_break()
    head("数据与预处理", 1)
    p("THINGS-EEG2 包含 10 名被试。每人有 1,654 个训练类，每类 10 张图像，以及 200 个未见测试类，每类 1 张图像。"
      "使用公开的 17 电极、100 Hz 预处理数据。下图取 Sub01 的第一张训练图像，展示已发布预处理文件中的单次响应与重复平均。"
      "纵轴保留文件数值单位。上游预处理后的幅度不能解释为原始传感器电压。")
    pic("图1_单次与重复平均", "图 1　Oz 电极的首次呈现和四次平均。虚线标记刺激开始。")
    pic("图2_电极时间响应", "图 2　同一图像四次平均后的电极响应。共用颜色范围便于比较各电极。")
    p("编码器取刺激后 0 至 0.8 秒（右端不含），得到 17 × 80 的输入。训练时同图 4 次平均，测试时 80 次平均。"
      "每个电极的均值和标准差在当前训练划分中估计，应用于相应验证或测试数据。")
    doc.add_page_break()
    head("模型与训练", 1)
    p("不同图像可在不同潜伏期引发响应。A 使用并行的短、中、长时间卷积核。不同刺激的有效电极也可能不同，B 在电极融合前为每个样本计算电极权重。"
      "四组保留相同的总时间特征宽度和视觉表示。基线使用单一时间尺度，A 和 B 分别单独加入对应改动，A+B 同时启用。")
    table(2, "信号与模型特征", ("特征", "构成", "含义"), [
        ("EEG 输入", "17 个电极 × 80 个时间点", "单幅图像的重复平均响应"),
        ("局部时间特征", "基线 11 点卷积；A 为 5、11、21 点卷积，各输出 8、16、8 维", "不同持续时间的局部变化"),
        ("电极权重", "特征和时间维取平均，17→8→17 维门控后重标定", "电极对该样本的相对贡献"),
        ("跨模态表示", "空间融合并投影为 768 维向量", "与图像模板进行点积比较")])
    p("训练时分别将 EEG 向量与配对 CLIP 向量归一化，计算 Sᵢⱼ = eᵢ · vⱼ / 0.07。"
      "以同批次的配对索引为标签，逐行和逐列求交叉熵，取两者的算术平均作为损失。"
      "开发集从训练侧按类留出 200 类，每类 5 张图像提供类别模板，其余 5 张提供 EEG 查询。"
      "依据开发验证确定共同轮数，再用全部训练类从头训练。")
    p(f"正式训练使用同一批量 {rows[0]['batch_size']}、同一轮数 {rows[0]['epochs']} 和固定种子 0–4。"
      "正式评分取单位化 EEG 表征与发布的 200 个原始中心模板直接点积，按每条查询的 200 个分数从高到低排名。")
    pic("图3_开发训练与验证", "图 3　Sub01、种子 0、A+B 的开发损失和留出类验证准确率。")
    doc.add_page_break()
    head("正式评估与计算成本", 1)
    p("对每个种子等权平均 10 名被试的成绩，再计算五个均值的平均和样本标准差。每组 50 次训练，耗时为训练时间总和，显存为单次峰值中的最大值。")
    table(3, "四组正式结果", ("模型", "Top-1 %", "Top-5 %", "参数量", "训练 h", "显存 MiB"),
          [(NAMES[r["config"]], f"{r['top1_pct']:.2f} ± {r['top1_seed_sd_pp']:.2f}",
            f"{r['top5_pct']:.2f} ± {r['top5_seed_sd_pp']:.2f}", str(r["parameters"]),
            f"{r['train_hours']:.2f}", f"{r['peak_gpu_mib']:.0f}") for r in overall])
    pic("图4_EEG与候选模板相似度", "图 4　Sub01、种子 0、A+B 的正式匹配得分。对角线对应真实类别。", 12.5)
    p("200 类均匀随机排序的 Top-1 和 Top-5 期望分别为 0.5% 和 2.5%。每条测试 EEG 的真实类别名次由对应得分矩阵重新计算。")
    doc.add_page_break()
    head("被试差异与配对消融", 1)
    by_subject = {(r["config"], r["subject"]): r for r in subjects}
    table(4, "逐被试五种子均值", ("被试", "基线 1/5 %", "仅 A 1/5 %", "仅 B 1/5 %", "A+B 1/5 %"),
          [(f"Sub{s:02d}", *(f"{by_subject[c,s]['top1_pct']:.1f} / {by_subject[c,s]['top5_pct']:.1f}"
                              for c in CONFIGS)) for s in SUBJECTS])
    effects = []
    for comparison, label in (("ab-b", "加入 A"), ("ab-a", "加入 B")):
        sub = [r for r in deltas if r["comparison"] == comparison]
        item = [label]
        for k in ("top1_delta_pp", "top5_delta_pp"):
            per = [mean(r[k] for r in sub if r["subject"] == s) for s in SUBJECTS]
            item.extend([f"{mean(per):+.2f}", f"{sum(d > 1e-10 for d in per)}/10"])
        effects.append(item)
    table(5, "成对消融", ("分量", "Top-1 变化 pp", "获益人数", "Top-5 变化 pp", "获益人数 "), effects)
    p(f"在 B 已启用时加入 A，Top-1、Top-5 分别改变 {effects[0][1]}、{effects[0][3]} 个百分点。"
      f"在 A 已启用时加入 B，变化分别为 {effects[1][1]}、{effects[1][3]} 个百分点。"
      "每个差值来自同一被试、同一种子的模型对，比对其他运行会同时混入被试差异或初始化差异。")
    p("17 通道、100 Hz 的输入与目标论文的传感器覆盖和时间分辨率不同。"
      "本实验检验两个模块在所用数据和训练设置上的贡献。权重变化还受模型容量及四次平均后的噪声影响，不能仅凭消融确定神经机制。")
    doc.add_page_break()
    head("调试与复现", 1)
    if args.debug_notes:
        for line in args.debug_notes.read_text(encoding="utf-8-sig").strip().splitlines():
            if line.strip():
                p(line)
    p("评测单元测试设三条查询，正确类别名次为 1、5、6，预期 Top-1 为 1/3，Top-5 为 2/3。"
      "将模板及其标签同步置换后应得到相同结果。复算程序同时核对正式结果文件中的实验键、准确率、训练日志及输出维度。")
    for i, source in enumerate(args.debug_image, start=5):
        dst = out / f"图{i}_调试记录{source.suffix.lower()}"
        shutil.copyfile(source, dst)
        body = p()
        body.paragraph_format.keep_with_next = True
        body.add_run().add_picture(str(dst), width=Cm(14.5))
        p(f"图 {i}　调试过程记录。", style="Caption")
    p("每条运行保存训练日志、权重、配置及 200 × 200 得分。根据运行说明可从权重重新评估，随后执行 "
      "eeg_evaluate.py 复算全部准确率。逐种子成绩、逐被试成绩及每一对消融差保存在独立 CSV 中。")
    head("参考文献", 1)
    for r in papers:
        p(f"[{r['id']}] {r['citation']} {r['source']}")
    doc.save(args.out / "实验报告.docx")
