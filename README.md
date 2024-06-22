# Pytorch 实验框架
## 0. 项目简介
本项目是一个基于Pytorch的实验框架，
在进行实验时，我们会经常性对模型内的模块进行调整，比如增加一个新的模块，或者修改一个模块的参数等等。
但这种修改的结果是不可预测的，有可能会导致模型的性能下降，也有可能会导致模型的性能提升。
因此，我们需要一个实验框架，来帮助我们更好的管理实验过程，以及实验结果。记录实验改动，实验结果，实验参数等等。
本项目就是一个基于Pytorch的实验框架，用于完善实验过程，提高实验效率。
参考大部分的开源代码和成熟的框架，本项目的主要采用配置文件的方式来管理实验参数，实验结果等，但减少了配置文件的复杂性，使实验的调试关注点集中在代码上而不是配置文件上。
## 1. 项目结构
```
.
├── data
│   ├── data_loader.py
│   ├── dataset.py
│   └── __init__.py
├── model
│   ├── model.py
│   └── __init__.py
├── README.md
├── train.py
└── utils
    ├── registry.py
    ├── logger.py
    └── __init__.py
```
## 2. 代码结构
- `data`文件夹：数据处理相关代码
