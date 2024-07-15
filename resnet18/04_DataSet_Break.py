## 此Python脚本在开发机上运行 ##

## Step 4

## 从训练集中抽样部分到验证集中，连带标签一同移动
## 通过WinSCP将所有图片存放在./DATASET_NMAE/train/image目录下
## 如果使用03标注，则将DATASET_NMAE改为03中一致的即可

DATASET_NMAE = "Dataset"  # 数据集名称
test_percent = 0.25  # 0.25表示25%的图片作为测试集

from random import sample
from shutil import move
from os import listdir

path = "./" + DATASET_NMAE + "/"

train_image = "train/image/"
train_label = "train/label/"
test_image = "test/image/"
test_label = "test/label/"

images_names = listdir(path + train_image)
# 抽样并移动
test_number = int(len(images_names)*test_percent)
test_names = sample(images_names, test_number)
for name in test_names:
    # 移动图片
    image_old = path + train_image + name
    image_path = path + test_image + name
    print(image_old, end=" ")
    try:
        move(image_old,image_path)
        print("\033[32;40m" + "Success." + "\033[0m")
    except:
        print("\033[31m" + "Failed! " + "\033[0m")
    
    # 移动标签
    label_old = path + train_label + name.split(".")[0] + ".txt"
    label_path = path + test_label + name.split(".")[0] + ".txt"
    print(label_old, end=" ")
    try:
        move(label_old, label_path)
        print("\033[32;40m" + "Success." + "\033[0m")
    except:
        print("\033[31m" + "Failed! " + "\033[0m")