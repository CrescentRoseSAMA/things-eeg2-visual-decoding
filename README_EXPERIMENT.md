# THINGS-EEG2 视觉解码实验运行说明

## 从实验材料包复算成绩

解压后进入含有 `eeg_train.py` 的项目根目录。所有命令均在该目录运行。Python 会将主脚本所在目录加入模块搜索路径。下面的自检应显示当前目录内的 `eeg_model.py`；如指向其他项目，请移除 `PYTHONPATH` 中对应的旧路径。

```powershell
python -c "import eeg_model; print(eeg_model.__file__)"
python verify_delivery.py --root . --device cpu
```

CPU 复算入口读取测试缓存、200 个固定候选模板、每次训练保存的标准化参数和模型权重。首轮 4 种配置共 200 次正式运行，复验通过后锁定的 4 点池化配置另有 50 次。首轮结果位于 `outputs/evaluation/`，后续正式评测和逐人逐种子配对明细位于 `outputs/pooling_formal/`。两次正式评测沿用相同测试类别。

单独运行 `python eeg_evaluate.py` 仅需 NumPy，可从首轮保存的得分复算统计表。运行 `python eeg_train.py evaluate --config ab --subject 1 --seed 0` 可重评首轮单个权重并更新该组的得分文件。

## 环境

在 64 位 Python 3.11 环境中安装下列依赖。本机训练栈和硬件见 `experiment_environment.json`。GPU 训练用 CUDA 11.8 版 PyTorch；运行报告生成器额外需要 python-docx、Matplotlib 和可显示中文的字体。Word 排版使用宋体和 Times New Roman。

```powershell
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-experiment.txt
python -c "import torch,numpy,docx; print(torch.__version__,torch.cuda.is_available(),numpy.__version__,docx.__version__)"
python -m unittest discover -s tests
```

## 数据和预处理

提交包内的 `data/THINGS-EEG2/derived/` 包含 10 名被试的训练及测试均值、冻结的图像特征与模板、开发划分和逐行样本索引。仅复算指标或从缓存训练时可以直接使用这些文件。若要重建全部缓存，请按以下步骤准备原始输入。

1. 用 Python **3.11 或更高版本**运行 `python _work/download_things_eeg2.py`。下载器使用标准库 `hashlib.file_digest`，核验发布方 SHA-256，下载 10 个公开档案（约 10.7 GB），解包到 `data/THINGS-EEG2/Preprocessed_data/sub-XX/`（另约 11.2 GB）。训练脚本环境与下载器解释器可以独立指定。压缩包及解包的全重复 EEG 档案由下载器获取。
2. 把发布的 `center_clip.npy`、`clip_feature_maps_training.npy`、`clip_feature_maps_test.npy` 放入 `data/THINGS-EEG2/dnn_feature/`。将公开刺激元数据保存为 `data/THINGS-EEG2/image_metadata.npy`。缓存构建时会核对四份文件哈希。[NICE 图像特征](https://github.com/eeyhsong/NICE-EEG/tree/main/dnn_feature) · [NEMAR 刺激元数据下载](https://nemar.s3.us-east-2.amazonaws.com/nm000232/objects/SHA256E-s656579--681a75633c59a356a66ca19a277429c1d48a00a98405ff96e53ee0b9f00b397c.npy)
3. 运行 `python eeg_data.py` 重建全部缓存；仅重建 Sub01 可用 `python eeg_data.py --subjects 1`。样本表记录刺激 ID、类别、EEG/图像行索引和开发划分角色。这里的单试次波形来自公开的预处理 EEG。

在 17 通道、100 Hz 发布数据上，代码截取刺激后 80 个点并分别对同图训练 4 次、测试 80 次取均值。每名被试的标准化参数由其当前训练划分拟合。上游公开的 MVNN 实现会在每个采集 session 内对训练与测试分区的协方差作平均后白化。因此，使用该版输入得到的结果需区分发布方预处理与本实验的训练、选模流程。[公开预处理代码](https://github.com/gifale95/eeg_encoding/blob/main/02_eeg_preprocessing/preprocessing_utils.py#L125-L166)

## 从头训练

在新目录中保留代码和 `data/THINGS-EEG2/derived/` 缓存，依次运行以下命令。阶段完成后再启动下一条。原目录重复运行会复用已有完成记录或恢复断点。

```powershell
python eeg_train.py smoke
python eeg_train.py verify-smoke
python eeg_train.py develop
python eeg_train.py formal
python eeg_train.py evaluate
python eeg_evaluate.py
```

`smoke` 在 Sub01 的 32 个配对上检查训练闭环；`verify-smoke` 经过权重重新加载检查检索排名。`develop` 对两个被试跑四种配置，以固定验证集选择共同训练轮数。`formal` 按四种配置、十名被试和五个种子训练，`evaluate` 使用发布的模板做点积评分。结果目录为 `outputs/runs/<配置>/sub-XX/seed-N/`，其中 `checkpoint.pt` 含模型、配置和标准化参数，`train.csv` 是逐轮日志，`scores.npy` 是逐查询得分，`metrics.json` 含耗时、显存及准确率。只有项目自己生成的可信权重才应交给 `torch.load`。

## 报告与提交

开发阶段的岭回归对照与 Sub01、Sub02 池化比较位于 `outputs/development_time_resolution/`，Sub03–Sub10、两个种子的三组配对复验位于 `outputs/pooling_replication/`。开发模型的权重和日志在 `outputs/dev/`。复现实验可在根目录按顺序运行以下命令，复用已有缓存及训练记录。

```powershell
python eeg_time_resolution.py
python eeg_pooling_dev.py
python eeg_pooling_replication.py
python eeg_pooling_formal.py
```

图表与独立源数据放在 `outputs/delivery_evidence/`。打包器将它们与代码、全部正式权重及日志、开发结果、缓存和调试记录一同归档。可在 Word 报告定稿前核验材料并生成实验材料包。

```powershell
python package_experiment.py --evidence-only --check-only
python package_experiment.py --evidence-only
```

实验材料包的默认路径是 `dist/THINGS-EEG2_实验材料.zip`。开发复验的手动检查点 ZIP 已在原工作区保存，包内保留其各项模型、运行数据与日志，避免重复占用空间。

将下方封面参数替换为本人信息。调试记录和截图可通过 `--debug-notes`、`--debug-image` 指定，保存实际报错与修改过程。

```powershell
python eeg_evaluate.py --report --college "学院" --class-name "班级" --student-id "学号" --name "姓名"
python package_experiment.py --check-only
python package_experiment.py --student-id "学号" --name "姓名"
```

上述报告生成器生成首轮四组对照的草稿。定稿时需补入后续 50 次正式成绩、配对比较、真实调试记录与对应编号图表。定稿文件路径为 `outputs/evaluation/实验报告.docx`，编号图表及源数据另存至同级 `图表/`。默认打包模式检查报告文件、正式运行、汇总表和图表，再在 `dist/` 生成按学号和姓名命名的 ZIP。提交电子包时按课程通知附纸质报告。
