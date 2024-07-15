## 此Python脚本在开发机上运行 ##

## Step 2

## 用于在同目录下，快速创建一个标准的数据集目录
## 通过WinSCP将01所有图片存放在./train/image目录下
## 运行此脚本后，会在当前目录下生成一个DataSet_3_1119文件夹，其中包含train和test两个文件夹
## 其中train文件夹包含image和label两个文件夹，test文件夹包含image和label两个文件夹
## 其中image文件夹包含所有图片，label文件夹包含对应的标签文件   


DATASET_NMAE = "Dataset"

from os import makedirs
path = "./" + DATASET_NMAE + "/"
try:
    makedirs(path + "train/image/")
    makedirs(path + "train/label/")
    makedirs(path + "test/image/")
    makedirs(path + "test/label/")
    print("Dirs Success")
except:
    print("Dirs Failed")