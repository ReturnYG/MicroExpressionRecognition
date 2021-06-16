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