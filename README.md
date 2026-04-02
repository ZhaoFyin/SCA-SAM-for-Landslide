
# SCA-SAM-for-Landslide

## 1 项目简介

本仓库用于遥感影像滑坡分割任务，主要实现了 **SCA-SAM** 模型及其对比实验、消融实验、测试评估与论文可视化绘图流程。仓库中包含主模型定义、数据集读取、损失函数、评价指标、对比模型训练、测试脚本，以及若干结果图绘制脚本。仓库根目录当前包含 `SAM`、`compare_model`、`cp`、`dataset`、`fig`、`losses`、`tools` 等目录，以及 `MySamModel.py`、`config.yaml`、`train_compare.py`、`train_sam.py`、`test.py`、`test_ablation.py` 和多个 `draw_*.py` 文件。 

---

## 2 仓库结构说明

```text
SCA-SAM-for-Landslide/
├── SAM/                     # 修改后的 SAM 主体代码
├── compare_model/           # 对比模型目录
├── cp/                      # 预训练权重与模型权重，网盘cp文件夹保存至此
├── dataset/                 # 数据集读取与变换
├── fig/                     # 绘制后的结果图保存目录
├── losses/                  # 损失函数
├── tools/                   # 指标计算工具
├── MySamModel.py            # SCA-SAM 主模型定义
├── config.yaml              # train_compare.py 使用的配置文件
├── draw_Ablation_fig.py     # 绘制消融实验结果图
├── draw_Channel_Map.py      # 绘制通道/特征响应图
├── draw_Compare_fig.py      # 绘制模型对比图
├── draw_Error_Cls.py        # 绘制错误分类分析图
├── draw_ZoomIn_fig.py       # 绘制局部放大图
├── test.py                  # 对比实验测试脚本
├── test_ablation.py         # 消融实验测试脚本
├── train_compare.py         # 对比模型训练脚本
├── train_sam.py             # SCA-SAM 训练脚本
├── utils.py                 # SCA-SAM 训练工具函数
└── utils_compare.py         # 对比实验训练工具函数
```

仓库根目录中可以直接看到上述文件与目录。


---

## 3 各个文件和目录的作用

### 3.1 核心模型文件

#### `MySamModel.py`
该文件定义了本项目的核心模型 **ScaSAM**。代码中通过 `sam_model_registry["vit_b"]` 构建 SAM 主干网络，并加载 `./cp/sam_vit_b_01ec64.pth` 作为基础权重；同时根据配置控制 SCA 模块与 LoRA 相关参数的训练方式，因此这是 SCA-SAM 的主体实现文件。 

---

### 3.2 训练脚本

#### `train_sam.py`
该文件用于训练 **SCA-SAM 主模型**。  
主要功能包括：

- 构建 `ScaSAM`
- 设置训练参数
- 构建训练集与验证集 DataLoader
- 调用 `utils.py` 中的训练函数进行训练
- 将日志输出到实验目录中

代码中默认输出目录为 `./results`，并在主程序中对数据集进行循环训练，目前主循环里写的是 `for data_name in ["BJL"]`。这意味着如果要训练别的数据集，需要直接修改该文件中的数据集设置。 

#### `train_compare.py`
该文件用于训练 **对比模型**。  
主要功能包括：

- 读取根目录下的 `config.yaml`
- 根据 `config.yaml` 中的 `dataset_name` 和 `model_list` 进行训练
- 自动创建输出目录
- 构建训练集、验证集
- 调用 `utils_compare.py` 中的训练函数

代码中明确使用 `with open("./config.yaml", 'r', encoding='utf-8') as f:` 读取配置文件，并在主程序中 `for net in config['model_list']:` 依次训练配置文件里列出的模型。

---

### 3.3 测试脚本

#### `test.py`
用于对比实验的测试与可视化输出。用于加载训练好的模型，在测试集上计算分割指标并保存结果图。

#### `test_ablation.py`
用于消融实验测试。主要面向不同模块设置、不同训练策略下的模型评估。

这两个文件都位于仓库根目录中。 
---

### 3.4 工具文件

#### `utils.py`
SCA-SAM 的训练工具函数文件，通常包含训练循环、验证循环、日志记录和模型保存等功能。

#### `utils_compare.py`
对比模型训练使用的工具函数文件，与 `train_compare.py` 配合使用。

---

### 3.5 数据与评价相关目录

#### `dataset/`
用于存放数据集读取脚本和数据预处理逻辑。`train_compare.py` 中在数据集为 `YYL` 或 `BJL` 时，会从 `dataset.landslide_dataset` 导入 `LandslideDataset`。说明滑坡数据集的核心读取逻辑就在该目录中。 

#### `tools/`
用于存放评价指标、工具类等辅助代码。

#### `losses/`
用于存放训练所需的损失函数定义。

---

### 3.6 对比模型目录

#### `compare_model/`
该目录用于存放所有对比模型。根据 `train_compare.py` 中的分支逻辑，目前支持的模型名称包括：

- `UNetFormer`
- `SegFormer`
- `SamLST`
- `EfficientViTSAM`
- `DCSwin`

因此，如果要做对比实验，需要在 `config.yaml` 的 `model_list` 中填写这些名称。

---

### 3.7 权重目录

#### `cp/`
该目录用于存放预训练权重和相关模型参数。`MySamModel.py` 中会读取 `./cp/sam_vit_b_01ec64.pth`，因此在运行前需要确保权重文件放置正确。网盘下载的cp文件夹放在此处。https://pan.baidu.com/s/1Y2vW2Ow5iK5iveMIo-Rg3g?pwd=9999

---

## 4 关于 draw 相关文件和结果图说明

### 4.1 `draw_Compare_fig.py`
该脚本用于绘制模型对比结果图。例如：

- `fig/FIG_BJL.png`
- `fig/FIG_YYL.png`

这类图一般用于展示不同模型在同一数据集上的分割结果可视化对比。 

### 4.2 `draw_ZoomIn_fig.py`
该脚本用于绘制局部区域放大对比图。例如：

- `fig/Zoom_in.png`

因此这个脚本对应的图通常用于展示滑坡局部边界、细节区域的放大结果。 

### 4.3 `draw_Ablation_fig.py`
该脚本用于绘制消融实验的可视化放大图。例如：

- `fig/Ablation_Zoom_in.png`

脚本主函数中还直接给出了 `BJL` 和 `YYL` 的若干样本编号与放大区域设置，因此它主要用于展示不同消融设置在典型样本上的分割差异。 

### 4.4 `draw_Channel_Map.py`
该脚本用于绘制特征响应图或通道响应图，例如：

- `fig/Channel_Map.png`

通常可用于展示不同阶段或不同模块下的特征激活情况。 

### 4.5 `draw_Error_Cls.py`
该脚本用于绘制错误分类分析图，例如：

- `fig/Error_Cls.png`

这类结果图适合用于分析模型错误区域、误检漏检情况或者类别混淆问题。 

---

## 5 数据集组织方式

对滑坡数据集而言，建议采用 VOC 风格的目录组织形式，例如：

```text
VOCdevkit_BJL/
└── VOC_landslide/
    ├── JPEGImages/
    ├── SegmentationObject/
    └── ImageSets/
        └── Segmentation/
            ├── train.txt
            ├── val.txt
            └── test.txt
```

---

## 6 如何运行

### 6.1 训练 SCA-SAM
修改train_sam.py中的数据集地址。随后：

```bash
python train_sam.py
```

该脚本默认将结果输出到 `./results` 目录下，并按数据集名称与时间戳建立实验文件夹。 

### 6.2 训练对比模型

修改`train_compare.py`、`config.yaml`中的数据集路径后，

```bash
python train_compare.py
```

该脚本会先读取根目录下的 `config.yaml`，然后按 `model_list` 中的模型顺序依次训练。

 `config.yaml` 内容如下：

```yaml
dataset_name: "BJL"
save_dir: "./compare_result"
uavid_dir: "./data/uavid_255"
vaihingen_dir: "./data/vaihingen"
potsdam_dir: "./data/potsdam"
bjl_dir: "C:/Users/48188/Data/VOCdevkit_BJL"
yyl_dir: "C:/Users/48188/Data/VOCdevkit_YYL"
model_list:
  - "SegFormer"
```

如果需要添加新模型组请添加在model_list中，如

```yaml
model_list:
  - "SegFormer"
  - "DCSwin"
```
这说明当前默认设置是在 `BJL` 数据集上按顺序训练 `SegFormer`、`DCSwin`，结果保存到 `./compare_result`。 

默认的结果文件夹名称为当前时间，训练后修改文件夹名称为本实验组名称。

对比训练组名称如`UNetFormer`, `SegFormer`, `DCSwin`, `SAM_Frozen`, `EfficientViTSAM`, `SAM_LST`, `SCA_SAM`

消融训练组名称如`Full_Frozen`, `SCA_SAM`, `Full_Train`,  `WO_SCA`


### 6.3 测试

修改`test.py`、`test_ablation.py`中数据集路径后，

```bash
python test.py
python test_ablation.py
```

### 6.4 绘图

修改数据集路径后，

```bash
python draw_Compare_fig.py
python draw_ZoomIn_fig.py
python draw_Ablation_fig.py
python draw_Channel_Map.py
python draw_Error_Cls.py
```

运行后生成的图片会保存在 `fig/` 目录中。

---

