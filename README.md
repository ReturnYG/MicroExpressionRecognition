# 项目说明

本项目主要包括微表情识别相关技术的代码实现，目前使用CASME、CASMEII、SAMM、SMIC数据集进行微表情识别实验，后续会增加更多数据集。
项目创建时间：2021年5月25日

## v0.1 2021.5.25

1.CASME、CASMEII、SAMNM数据集的预处理 \
2.SMEConvNet的实现

## v0.2 2021.6.16
1.增加了SMIC数据集的预处理 \
2.抽取了预处理中面部提取的部分以简化代码 \
3.增加了面部水平翻转的代码以扩增数据集 \
4.从keras官方copy了部分网络代码以进行实验\
To Do: 完善注释、解决目前存在的过拟合问题。

## v1.0 2022.7.8
1、增加了不同方式的预处理，包括光流法、LOO方式等处理。并且是否遮盖面部、是否矫正面部位置等皆可在预处理时进行选择。\
2、在utils文件夹中增加了部分工具。\
3、进一步完善注释。

项目结构如下： 
```
├── additional OpenCV及Dlib所需文件
├── data_save 预处理后存放数据集
│   ├── new LOSO并不完全处理
│   ├── new_all LOSO并完全处理
│   ├── new_eyemask LOSO仅包含眼部遮罩
│   ├── new_withFaceMask LOSO仅包含面部遮罩
│   └── old 旧处理
├── datasets 预处理代码
│   ├── casmeDatasets.py CASME数据集（并未实际使用）
│   ├── casmeiiDatasets.py CASME II数据集
│   ├── sammDatasets.py SAMM数据集
│   └── smicDatasets.py SMIC数据集
├── models 训练模型
│   ├── CBAM_module_v2.py
│   ├── CBA_module.py
│   ├── Confusion Matrix Image 保存混淆矩阵
│   │   ├── Test 测试
│   │   ├── Train 训练
│   │   └── result 最终模型
│   ├── MobileNetV3.py
│   ├── MobileVit_Keras.py
│   ├── ShuffleNetV2.py
│   ├── checkpoint tensorflow检查点，
│   │   ├── assets
│   │   ├── saved_model.pb
│   │   └── variables
│   ├── error_model 准确率出现异常时保存模型及权重方便排错
│   ├── log tensorflow包含的日志
│   ├── save_weight 保存准确率最高时的模型权重
│   ├── train.py 部分训练代码
│   └── train_11.1.py 部分训练代码
└── utils 
    ├── Grid_search.py 网格搜索确定部分超参数
    ├── batchSizeFix.py 根据输入的batch_size调整数据集列表
    ├── dataExpansion.py 数据扩增
    ├── dataPreparation.py 数据准备
    ├── faceAlignment.py 面部对齐
    ├── faceDealer.py 面部处理函数
    ├── faceExtraction.py 面部提取
    ├── facemask.py 面部遮罩
    ├── frameNormalized.py 帧归一化
    ├── opticalFlowTVL1.py 提取TVL1光流
    └── sampleBalance.py 样本数量平衡
```

部分依赖：

python = 3.8

numpy >= 1.18.5

opencv-python >= 4.5

scikit-learn >=0.24.2

keras >= 2.2.4

dlib >= 19.21.1

pandas >= 1.2.4

tensroflow = 0.1a0(经Apple修改的版本)


