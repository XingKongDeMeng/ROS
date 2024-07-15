import json
import os
import shutil
import sys
import argparse
import random

source_folder_path = "train"
target_folder_path = "line_follow_dataset"

train_ratio = 0.9

parser = argparse.ArgumentParser()
parser.add_argument("--source_folder_path", type=str, help="")
parser.add_argument("--target_folder_path", type=str, help="")
args = parser.parse_args()
if args.source_folder_path:
    source_folder_path = args.source_folder_path
if args.target_folder_path:
    target_folder_path = args.target_folder_path

train_folder = os.path.join(target_folder_path, 'train')
train_image = os.path.join(train_folder, 'image')
train_label = os.path.join(train_folder, 'label')

test_folder = os.path.join(target_folder_path, 'test')
test_image = os.path.join(test_folder, 'image')
test_label = os.path.join(test_folder, 'label')

if not os.path.exists(target_folder_path):
    os.makedirs(target_folder_path)

if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(train_image):
    os.makedirs(train_image)
if not os.path.exists(train_label):
    os.makedirs(train_label)

if not os.path.exists(test_folder):
    os.makedirs(test_folder)
if not os.path.exists(test_image):
    os.makedirs(test_image)
if not os.path.exists(test_label):
    os.makedirs(test_label)

# 获取所有文件的路径
all_files = []
for filename in os.listdir(source_folder_path):
    if filename.endswith(".json"):
        json_path = os.path.join(source_folder_path, filename)
        base_name = os.path.splitext(filename)[0]
        jpg_path = os.path.join(source_folder_path, f"{base_name}.jpg")
        if os.path.exists(jpg_path):
            all_files.append((json_path, jpg_path))

# 打乱文件顺序
random.shuffle(all_files)

# 计算训练集和测试集的大小
num_files = len(all_files)
num_train = int(train_ratio * num_files)
train_files = all_files[:num_train]
test_files = all_files[num_train:]

def process_files(files, image_folder, label_folder):
    for json_path, jpg_path in files:
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        with open(json_path, "r") as json_file:
            data = json.load(json_file)
            shapes = data["shapes"]
            num_points = 0
            for shape in shapes:
                shape_type = shape["shape_type"]
                if shape_type == "point":
                    num_points += 1
                    if num_points == 2:
                        print("Two points appear")
                        print("Please check the file " + json_path)
                        sys.exit()
                    points = shape["points"]
                    for point in points:
                        x, y = point
                        shutil.copy(jpg_path, image_folder)
                        with open(os.path.join(label_folder, f"{base_name}.txt"), 'w') as txt_file:
                            txt_file.write(f"{int(x)} {int(y)}")
                        print(f"Copied: {jpg_path} --> {image_folder}")

# 处理训练集
process_files(train_files, train_image, train_label)

# 处理测试集
process_files(test_files, test_image, test_label)
