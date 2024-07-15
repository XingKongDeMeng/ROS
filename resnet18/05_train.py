## 此Python脚本在开发机上运行 ##

# Step 5

# 训练ResNet18
# 如果是第一次训练会自动下载预训练权重，约40MB
# 训练结束后会在当前目录下生成一个名为BEST_MODEL_PATH的模型文件
# CPU就能训练，我的R7-4800H约12秒一个Epoch，不会太慢

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import glob
import PIL.Image
import os
import numpy as np
from threading import Thread
from time import time, sleep

DATASET_NMAE = "Dataset"   # 数据集名称
BEST_MODEL_PATH = './model_best.pth'  # 最好的训练结果
BATCH_SIZE = 64
NUM_EPOCHS = 100            # 迭代次数


def main(args=None):
    best_loss = 1e9
    train_image = "./" + DATASET_NMAE + "/train/"
    test_image = "./" + DATASET_NMAE + "/test/"
    train_dataset = XYDataset(train_image)
    test_dataset = XYDataset(test_image)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # 创建ResNet18模型，这里选用已经预训练的模型，
    # 更改fc输出为2，即x、y坐标值
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 2)
    device = torch.device('cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(NUM_EPOCHS):
        epoch_time_begin = time()
        model.train()
        train_loss = 0.0
        for images, labels in iter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.mse_loss(outputs, labels)
            train_loss += float(loss)
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)

        model.eval()
        test_loss = 0.0
        for images, labels in iter(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = F.mse_loss(outputs, labels)
            test_loss += float(loss)
        test_loss /= len(test_loader)
        msgStr = "Epoch" + "\033[32;40m" + " %d " % epoch + "\033[0m"
        msgStr += "-> time: \033[32;40m%.3f\033[0m s,  train_loss: \033[32;40m%f\033[0m,  test_loss: \033[32;40m%f\033[0m" % (
            time() - epoch_time_begin, train_loss, test_loss)

        if test_loss < best_loss:
            msgStr += (" \033[31m" + " Saved" + "\033[0m")
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_loss = test_loss
        else:
            msgStr += " Done"
        print(msgStr)


class XYDataset(torch.utils.data.Dataset):
    def __init__(self, directory, random_hflips=False):
        self.directory = directory
        self.random_hflips = random_hflips
        self.image_paths = glob.glob(os.path.join(
            self.directory + "/image", '*.jpg'))
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = PIL.Image.open(image_path)
        with open(os.path.join(self.directory + "/label", os.path.splitext(os.path.basename(image_path))[0]+".txt"), 'r') as label_file:
            content = label_file.read()
            values = content.split()
            if len(values) == 2:
                value1 = float(values[0])
                value2 = float(values[1])
            else:
                print("文件格式不正确")
        x, y = value1, value2

        if self.random_hflips:
            if float(np.random.rand(1)) > 0.5:
                image = transforms.functional.hflip(image)
                x = -x

        image = self.color_jitter(image)
        image = transforms.functional.resize(image, (224, 224))
        image = transforms.functional.to_tensor(image)
        image = image.numpy().copy()
        image = torch.from_numpy(image)
        image = transforms.functional.normalize(image,
                                                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image, torch.tensor([x, y]).float()


if __name__ == '__main__':
    main()
