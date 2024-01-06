# 大数据机器学习大作业

## 项目简介
见`大实验.pdf`

## 数据准备
在[这里](https://drive.google.com/file/d/13eZSqjGYR3cisCETFuhKmzHuBc_gtgsq/view?usp=sharing)下载数据集，解压后放在`PACS`文件夹下。

## 环境配置
```
conda create -f environment.yml
```

## checkpoint读取
如果想直接进行测试，可以在[这里](https://pan.baidu.com/s/1WAq043Nla5SSkYdKgZTPLg)获取训练好的模型（提取码为1234），放在`Checkpoints`文件夹下。

## 训练
```
sh train.sh
```

## 测试
```
sh test.sh
```

## 结果
结果最终保存在`result.csv`中。
