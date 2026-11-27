# 北京理工大学《数据挖掘技术》专题作业报告（物料草稿）

> **排版提示（按模板要求）**：最终导出为 Word（`.docx`）并打印纸质版时，请将封面填写完整（学院、班级、姓名、学号、年月日），正文统一设为**英文 `Times New Roman`、中文 `宋体`、字号 `5 号`、行距 `1.5 倍行距`**。

---

## 题目：EEG-视觉解码

### 1. 任务描述与模型选择依据及算法步骤

#### 1.1 任务描述
本专题研究基于脑电信号（EEG）的跨模态视觉解码任务（Zero-Shot EEG-to-Image Retrieval）：给定一名被试观看自然图像刺激时采集的 EEG 信号，在 **200 个训练阶段完全未见过的视觉类别**的固定 CLIP 图像特征模板中检索出对应的正确视觉类别。评测指标以 **Top-1 准确率** 和 **Top-5 准确率** 为主，并补充 **Macro-F1、Micro-F1 及 P-R 曲线（PR-AUC）**，课程对标参考线为 NICE-GA 被试内成绩（Top-1 **15.6%**，Top-5 **42.8%**）[1]。

#### 1.2 模型选择依据
1. **数据特性**：THINGS-EEG2 数据集具有“多通道空间拓扑（17 通道枕顶叶）+ 短时序动态（0–0.79 s，100 Hz，80 采样点）+ 高噪单试次重复”的特点[2]。纯展平 MLP（第 12 轮复现 `EEGProject`）在本数据集上仅取得开发集 Top-5 `20.925%`，显著低于卷积网络，说明保留二维时空结构与局部归纳偏置至关重要。
2. **编码器架构**：借鉴 **EEGNet**[4]、**EEG-Inception**[5] 与 **SENet**[6]，采用“多尺度时间卷积（核长 5/11/21）+ 电极门控注意力（SE）+ 空间导联卷积 + 四点时间池化 + 零初始化残差非线性投影头（`ab_pool4_resproj_cosine`）”，将 EEG 映射至 768 维 L2 归一化 CLIP 视觉语义空间[3]。
3. **训练与检索校正**：针对单试次信噪比低的问题，采用**在线试次子集平均**（每轮随机抽 2/3/4 次重复求均值）；针对视觉模板“枢纽点偏置（Hubness）”，在推理阶段引入 **QB-Norm（Querybank Normalisation）**[8]；针对“差一名进前五”的边界样本，在第 15 轮加入 **Top-5 间隔辅助损失**[9]。

#### 1.3 拟用算法步骤
1. **数据校验与切分**：校验 `image_metadata.npy` 与 CLIP 特征的 SHA-256 哈希；官方训练集 1,654 类（16,540 张图 × 4 重复）与测试集 200 类（200 张图 × 80 重复）严格类别互斥；开发集按固定种子 `2026` 划分 1,454 训练类与 200 验证类。
2. **通道标准化**：仅在当前划分的训练 EEG 上计算 17 通道各自的均值与标准差，执行逐通道 Z-score 标准化，防止测试集信息泄漏。
3. **特征提取与映射**：EEG 经多尺度时间卷积、SE 电极门控、空间卷积、四点时间池化与残差投影头，输出 768 维单位向量 $z \in \mathbb{R}^{768}$。
4. **联合损失训练**：使用对称 InfoNCE 对比损失、配对余弦回归损失与 Top-5 间隔辅助损失联合优化，Adam 学习率 `2e-4`，批大小 `128`，训练 `100` epochs。
5. **QB-Norm 推理检索**：利用同一被试的 16,540 条训练 EEG 作为查询库，计算动态倒置 softmax 校正相似度（$\beta=20, k=1$），对 200 个未见测试类模板排序并输出 Top-1/Top-5 命中与评价指标。

---

### 2. 数据描述、原始数据可视化与预处理说明

- **数据规模**：10 名被试（`sub-01` ~ `sub-10`），17 个枕顶叶视觉相关电极（`Pz, P3, P7, O1, Oz, O2, P4, P8, P1, P5, PO7, PO3, POz, PO4, PO8, P6, P2`），采样率 `100 Hz`，时间窗 `0.00–0.79 s`（每试次 `17 × 80`）。
- **原始数据可视化**：见 `Asset/图1_原始EEG单次与重复平均波形.png` 与 `Asset/图2_17通道电极时间响应特征图.png`。单次试次波动剧烈，经 4 次或 80 次重复平均后，枕顶叶视觉诱发电位（VEP）在 100–300 ms 区间呈现清晰波峰波谷。
- **预处理流程**（详见 `Asset/表1_数据预处理与划分汇总.csv`）：
  1. **缺值与完整性核验**：检查全部数组无 `NaN`/`Inf`，通过 SHA-256 校验保证 EEG 行号与图像特征行号一一对应。
  2. **滤波与时窗截取**：使用官方预处理后的 100 Hz 枕顶叶 17 通道数据，截取刺激后 0–0.79 s 有效响应窗。
  3. **噪声抑制（试次平均增强）**：测试时使用官方 80 次重复平均；训练时每 epoch 对每张图无放回抽取 2、3 或 4 条真实重复试次求均值，兼顾降噪与样本多样性。
  4. **离散化（时间槽池化）**：网络内部通过核长 `(1, 4)`、步长 `(1, 4)` 的平均池化将 80 个连续采样点离散压缩为 20 个 `40 ms` 时间槽。
  5. **归一化**：对 EEG 按训练集通道统计量做 Z-score 标准化；对编码器输出向量与 CLIP 模板做 L2 单位化。

---

### 3. 特征提取与物理含义

编码器结构见 `Asset/图3_EEG视觉解码编码器结构图.png`，完整特征提取表见 `Asset/表2_特征提取与物理含义.csv`，特征响应可视化见 `Asset/图2_17通道电极时间响应特征图.png`。

1. **多尺度时间卷积特征**：并联 3 组二维卷积核 $(1, 5)$、$(1, 11)$、$(1, 21)$（在 `100 Hz` 下分别对应 `50 ms`、`110 ms`、`210 ms` 感受野），分别捕获早期高频瞬态视觉响应、中期 P1/N1 成分与晚期语义加工慢波，输出拼接为 `32` 通道。
2. **电极门控注意力（SE）**：对时间和特征通道求全局均值得到 17 维电极描述向量 $s \in \mathbb{R}^{17}$：
   $$s_c = \frac{1}{F \cdot T}\sum_{f=1}^{F}\sum_{t=1}^{T} X_{f,c,t}, \qquad w = \sigma\!\left(W_2\,\mathrm{ReLU}(W_1 s)\right)$$
   其中 $W_1 \in \mathbb{R}^{8 \times 17}, W_2 \in \mathbb{R}^{17 \times 8}$，自适应加权枕顶叶有效导联。
3. **空间导联融合与四点池化**：以 `(17, 1)` 空间卷积整合全脑区通道信息后，经 `(1, 4)` 平均池化得到 `32 × 1 × 20` 特征图（展平为 640 维）。
4. **残差语义投影**：经 `Linear(640, 768)` 与零初始化瓶颈残差块（`LayerNorm → Linear(768,256) → GELU → Dropout(0.5) → Linear(256,768)`）映射并做 L2 归一化：
   $$z = \frac{h_0 + \mathrm{Res}(h_0)}{\|h_0 + \mathrm{Res}(h_0)\|_2}$$

---

### 4. 模型参数调优过程、计算原理与公式

#### 4.1 核心计算公式
1. **联合训练损失（第 5–15 轮）**：
   $$\mathcal{L} = 0.75\,\mathcal{L}_{\mathrm{InfoNCE}} + 0.25\,\left(1 - \frac{1}{B}\sum_{i=1}^{B} z_i^\top v_i\right) + 0.10\,\mathcal{L}_{\mathrm{top5}}$$
   其中对称 InfoNCE 温度固定为 $\tau = 0.07$：
   $$\mathcal{L}_{\mathrm{InfoNCE}} = \frac{1}{2}\left(\mathrm{CE}\!\left(\frac{ZV^\top}{\tau}, y\right) + \mathrm{CE}\!\left(\frac{VZ^\top}{\tau}, y\right)\right)$$
   Top-5 间隔辅助损失 $\mathcal{L}_{\mathrm{top5}}$ 从训练集随机抽取 200 类构建画廊（正类模板排除当前配对图像，减少泄漏），仅当第 5 高错误类相似度 $s_{\mathrm{neg}}^{(5)}$ 高于正类相似度 $s_{\mathrm{pos}}$ 时惩罚：
   $$\mathcal{L}_{\mathrm{top5}} = \frac{1}{B}\sum_{i=1}^{B}\mathrm{ReLU}\!\left(\frac{s_{\mathrm{neg}, i}^{(5)} - s_{\mathrm{pos}, i}}{0.07}\right)$$
2. **QB-Norm 推理校正（第 13–15 轮）**：
   利用同被试训练 EEG 查询库 $Q_{\mathrm{train}}$ 计算每个模板 $j$ 的归一化分母，在推理阶段对测试相似度向量 $c \in \mathbb{R}^{200}$ 执行动态倒置 softmax（$\beta=20, k=1$）：当最高分模板曾被训练库选为 Top-1 时启用校正，抑制高人气误选模板。

#### 4.2 参数调优与 1–15 轮迭代过程
迭代演进柱状图见 `Asset/图4_迭代优化里程碑对比图.png`，完整汇总表见 `Asset/表3_迭代里程碑正式评测汇总.csv`：
- **架构探索（基线 → `ab_pool4`）**：Top-1/Top-5 从 `10.17% / 31.91%` 提升至 `12.00% / 35.73%`。
- **第 3–5 轮（残差投影 + 联合损失 + CSLS）**：加入零初始化残差头与 `0.75 InfoNCE + 0.25 Cosine` 联合损失，并引入 CSLS（$k=10$），正式成绩跃升至 `14.70% / 39.01%`。
- **第 6 轮（在线试次子集平均）**：随机抽 2/3/4 重复求均值，正式成绩达 `16.64% / 40.71%`（Top-1 超越课程线）。
- **第 7–12、14 轮（开发集消融筛选）**：分别验证了 CLIP 软目标、跨被试预训练、分数融合、电极自注意力、单试次+高斯噪声+EMA 及 `EEGProject` 展平骨干，均未在独立开发集同时通过严谨门槛，故未盲目进入全量。
- **第 13 轮（QB-Norm 检索校正）**：复用第 6 轮权重，无新增训练，仅改用 QB-Norm（$\beta=20, k=1$），正式成绩提升至 `17.44% / 42.43%`。
- **第 15 轮（Top-5 间隔辅助损失 + QB-Norm）**：正式 50 组全量训练收官，Top-1 达 **17.86%**、Top-5 达 **43.66%**，代表性收敛曲线见 `Asset/图5_第15轮训练损失收敛曲线.png`。

---

### 5. 原代码风格、模块说明与变量命名规则

- **代码文件清单**：
  - `eeg_data.py`：数据完整性校验、SHA-256 核验、2026 开发划分与索引导出。
  - `eeg_model.py`：编码器（`ab_pool4_resproj_cosine`、`EEGProject` 等）、对称 InfoNCE、余弦损失与 `ClassPrototypes.top5_loss`。
  - `eeg_train.py`：Z-score 标准化、在线试次子集抽样、Adam 优化器、断点续训（`resume.pt`）与日志记录。
  - `eeg_querybank.py`：QB-Norm 查询库统计与动态倒置 softmax 评分。
  - `eeg_csls_replication.py`：CSLS（$k=10$）近邻偏置校正评分。
  - `eeg_top5_qb_pilot.py`：第 15 轮先导与 50 组全量训练评测主入口（`--full`）。
- **主要第三方模块**：`torch`（张量运算与自动微分）、`numpy`（内存映射与矩阵计算）、`matplotlib`（科研绘图）、`csv / json / pathlib / hashlib`（标准库数据流与校验）。
- **变量命名规则**：
  - 全大写常量表示全局超参数或路径（如 `EPOCHS = 100`、`BATCH = 128`、`LR = 2e-4`、`BETA = 20`、`TOP5_RANK_WEIGHT = 0.1`、`CHANNELS`）；
  - 下划线小写（snake_case）表示函数、张量与中间变量（如 `train_eeg`、`test_centers`、`qb_scores`、`mean`、`std`）；
  - 驼峰命名（PascalCase）表示 PyTorch 模型与类（如 `ClassPrototypes`、`ResidualProjector`）。

---

### 6. 调试分析（典型问题与修复记录）

> **截图提示**：请在本节补充 1–2 张你在终端运行报错与修复后正常运行时的真实截图。

1. **沙箱/多环境 Python 解释器启动异常（`0xc0000022`）**
   - **问题原因**：系统默认 `python` 指向 `D:\anacoanda\python.exe`，在受限环境下无法正常初始化 CUDA/DLL；需切换至 `D:\py\python.exe` 或指定环境 `D:\anacoanda\envs\yolov8\python.exe`。
   - **修复方法**：明确区分训练环境与分析环境，避免环境变量混用。
2. **全量 50 组训练耗时较长导致的中途中断与续训**
   - **问题原因**：第 15 轮每组训练需约 100–310 秒，50 组累计约 3 小时，在 `sub-07 / seed-4` 第 97 个 epoch 时曾发生进程中断。
   - **修复方法**：`eeg_train.py` 内置每轮保存 `resume.pt` 断点检查点与 `train.csv` 状态日志，重新启动 `eeg_top5_qb_pilot.py --full` 后自动从 `sub-07 / seed-4` 续训并顺利完成全部 50 组训练。
3. **测试集信息泄漏风险排查**
   - **问题原因**：若对全量数据统一做 Z-score 标准化或用测试样本估计模板偏置，会虚高检索分数。
   - **修复方法**：在 `eeg_data.py` 与 `eeg_train.py` 中强制仅用当前划分的训练集拟合 `mean`/`std`，并在测试前通过 `torch.testing.assert_close` 严格核对基线与候选的归一化统计量完全一致。

---

### 7. 模型评价（Acc、P-R 曲线、F1 指标）

基于 10 名被试 × 5 个种子共 **10,000 次**官方 200 类测试查询的完整 `qb_scores.npz` 矩阵计算得到评价指标（详见 `Asset/表4_模型评价指标汇总.csv` 与 `Asset/图6_模型评价PR曲线与F1对比.png`）：

| 方案 | Top-1 Acc (%) | Top-5 Acc (%) | Macro-F1 (%) | Micro-F1 (%) | 200类检索 PR-AUC (%) | Top-1置信度 PR-AUC (%) |
|---|---:|---:|---:|---:|---:|---:|
| 第 13 轮 (`round06_qb_norm`) | 17.44 | 42.43 | 15.30 | 17.44 | 11.15 | 32.85 |
| **第 15 轮 (`round06_top5_margin_qb`)** | **17.86** | **43.66** | **15.67** | **17.86** | **11.23** | **32.68** |
| 课程目标线 (`NICE-GA`) | 15.60 | 42.80 | — | — | — | — |

- 在 200 类单标签检索任务中，Top-1 准确率等价于 Micro-F1（`17.86%`）；Macro-F1 从 `15.30%` 提升至 `15.67%`，说明 Top-5 间隔损失与 QB-Norm 并未以牺牲冷门类别为代价，而是整体改善了 200 个未见类别的检索排序质量。

---

### 8. 运行结果与结论分析

- **总体结果**：第 15 轮全量评测取得 **Top-1 = 17.86%（±0.22 pp）**、**Top-5 = 43.66%（±0.60 pp）**，相对课程参考线（15.60% / 42.80%）分别高出 **+2.26** 和 **+0.86 个百分点**，是本项目**首次同时超越课程两项参考目标**的完整方案。
- **个体差异与排序结构**：分被试对比见 `Asset/图7_十名被试Top1与Top5准确率对比.png` 与 `Asset/表5_分被试五种子均值对比.csv`。10 名被试中 6 名 Top-1 提升、9 名 Top-5 提升（`sub-10` Top-5 达 `51.50%`，`sub-08` 达 `49.80%`）。从 `Asset/图8_EEG与200类模板QB-Norm相似度矩阵.png` 可见主对角线呈现明显的正匹配高响应带。
- **主要结论与局限**：
  1. 在 17 通道、100 Hz、0–0.79 s 的受限输入条件下，通过**在线试次子集平均 + 残差语义投影 + Top-5 间隔损失 + QB-Norm** 组合，能够超越 63 通道 250 Hz 的课程基线；
  2. 跨被试个体差异仍然显著（`sub-05` Top-1 约 `13.10%`，而 `sub-04` 达 `22.60%`），未来可进一步结合跨被试对齐或自适应滤波提升弱被试表现。

---

### 9. 提交文件说明（`学号+姓名` 文件夹结构）

- 电子版与纸质版实验报告（`.docx` / `.pdf`）；
- 可执行的全部 `*.py` 源代码（`eeg_data.py`、`eeg_model.py`、`eeg_train.py`、`eeg_querybank.py`、`eeg_csls_replication.py`、`eeg_top5_qb_pilot.py` 等）；
- 权重文件（`checkpoint.pt`）与过程/结果数据文件（`Asset/` 下全部 `图1`–`图8` 与 `表1`–`表5`）。

---

### 10. 参考文献列表

[1] Song Y., Liu B., Li X., et al. Decoding Natural Images from EEG for Object Recognition[C]//ICLR, 2024.  
[2] Gifford A. T., Dwivedi K., Roig G., Cichy R. M. A large and rich EEG dataset for modeling human visual object recognition[J]. NeuroImage, 2022, 264: 119754.  
[3] Radford A., Kim J. W., Hallacy C., et al. Learning Transferable Visual Models From Natural Language Supervision[C]//ICML, 2021: 8748–8763.  
[4] Lawhern V. J., Solon A. J., Waytowich N. R., et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of Neural Engineering, 2018, 15(5): 056013.  
[5] Santamaría-Vázquez E., Martínez-Cagigal V., Vaquerizo-Villar F., Hornero R. EEG-Inception: A Novel Deep Convolutional Neural Network for Assistive ERP-Based Brain-Computer Interfaces[J]. IEEE TNSRE, 2020, 28(12): 2773–2782.  
[6] Hu J., Shen L., Sun G. Squeeze-and-Excitation Networks[C]//CVPR, 2018: 7132–7141.  
[7] Conneau A., Lample G., Ranzato M., et al. Word Translation Without Parallel Data[C]//ICLR, 2018.  
[8] Bogolin S.-V., Croitoru I., Jin H., et al. Cross Modal Retrieval With Querybank Normalisation[C]//CVPR, 2022.  
[9] Berrada L., Zisserman A., Kumar M. P. Smooth Loss Functions for Deep Top-k Classification[C]//ICLR, 2018.  
[10] Li D., Wei C., Li S., et al. Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion[C]//NeurIPS, 2024.  
[11] Wu H., Li Q., Zhang C., et al. Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior[C]//CVPR, 2025.  
[12] Yu K., Huang C., Liu Z., Wu K. ProtoEEG: Structured Prototype Alignment for EEG Visual Decoding[J]. arXiv:2610.33065, 2026.  
[13] Wang Y., Jian X., Xue B. Balance Act: Mitigating Hubness in Cross-Modal Retrieval with Query and Gallery Banks[C]//EMNLP, 2023.  
[14] Chen H., et al. Visual Neural Decoding via Improved Visual-EEG Semantic Consistency[J]. arXiv:2408.06788, 2024.  
[15] Cui Z., et al. SCORE: Subject Coordinate Recovery for Label-Free Cross-Subject EEG-to-Image Retrieval[J]. arXiv:2608.19134, 2026.  
[16] Wang J., et al. NeuroCLIP: Brain-Inspired Prompt Tuning for EEG-to-Image Multimodal Contrastive Learning[J]. arXiv:2511.09250, 2025.  
[17] Wang J., Chen K. CORTIVA: Candidate-Score Fusion of Complementary Visual Teachers for EEG- and MEG-to-Image Retrieval[J]. arXiv:2608.01355, 2026.
