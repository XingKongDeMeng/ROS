## 此Python脚本在开发机上运行 ##

## Step 3

## 用于标注数据集
## 通过WinSCP将01采集到的所有图片存放在./train/image目录下
## 生产的标签txt文件存放在./train/label目录下
## 直接存储为8位有效数字的浮点数，例如(0.22023810, 0.85267857), 照片长和宽在10^8以下不会产生精度问题


import cv2
import os

DATASET_NMAE = "Dataset"  # 数据集名称

#一般1080p屏幕取2.0为佳，2K屏幕取3.0为佳，4K屏幕取4.0为佳。由于标签直接使用浮点数存储，故不会因缩放产生坐标变换的问题。
ZOOM = 2.0  # 显示缩放倍数，与标注数据无关，仅仅适应一些高分屏的电脑

def get_xy(file_path):
    # 读取txt文件，获取x,y坐标(浮点数表示)
    x,y = -1.0,-1.0
    with open(file_path) as f:
        content = f.read().split(" ")
        x, y = float(content[0]), float(content[1])
        f.close()
    return x,y

def mouse_callback(event, x, y, flags, param):
    # 鼠标点击事件
    global img_x, img_y, label_path, txt_name,img,img_width, img_height
    if event == cv2.EVENT_LBUTTONUP:
        img_x, img_y = float(x)/img_width/ZOOM, float(y)/img_height/ZOOM
        cv2.imshow("img", cv2.circle(img.copy(), (x, y), 10,(0,0,255), -1))
        print("Mouse Click(%d, %d), Save as(%.8f, %.8f)"%(x,y,img_x,img_y))
        with open(label_path + txt_name,"w") as f:
            f.write("%.8f %.8f"%(img_x, img_y))


# 新建cv2的工作窗口，并绑定鼠标点击的回调函数
img_x, img_y = -1,-1
cv2.namedWindow('img')
cv2.setMouseCallback('img', mouse_callback)

img_path = DATASET_NMAE + "/train/image/"
label_path = DATASET_NMAE + "/train/label/"
print("img path = %s"%img_path)
print("label path = %s"%label_path)

img_names = os.listdir(img_path)

# img size
img_width, img_height = 0, 0      
# img control
img_control = 0
img_control_min = 0
img_control_max = len(img_names) - 1
while True:
    name = img_names[img_control]
    print(name, end="  ")
    img = cv2.imread(img_path + name)
    img_height, img_width = img.shape[:2]
    img = cv2.resize(img, (0,0), fx=ZOOM, fy=ZOOM)
    cv2.imshow("img", img)
    print("height = %d, width = %d"%(img_height, img_width), end="  ")
    
    ## 若存在标签则绘制点，若不存在则不绘制
    txt_name = name.split(".")[0] + ".txt"
    label_names = os.listdir(label_path)
    if txt_name in label_names:
        img_x, img_y = get_xy(label_path + txt_name)
        cv2.imshow("img", cv2.circle(img.copy(), (int(ZOOM*img_width*img_x), int(ZOOM*img_height*img_y)), 10,(0,0,255), -1))
        # print(int(ZOOM*img_width*img_x), int(ZOOM*img_height*img_y))
        print("\033[32;40m" + "Label Exist" + "\033[0m" + ": x = %.8f, y = %.8f"%(img_x, img_y))
    else:
        print("\033[31m" + "NO Label" + "\033[0m")
    

    ## while 循环的控制
    command = cv2.waitKey(0) & 0xFF
    # 慢速退
    if command == ord('a'):
        if img_control > img_control_min:
            img_control -= 1
        else:
            img_control = 0
            print("First img already")
    # 慢速进
    elif command == ord('d'):
        if img_control < img_control_max:
            img_control += 1
        else:
            img_control = img_control_max
            print("Last img already")
    # 快速退
    elif command == ord('z'):
        if img_control - 4 > img_control_min:
            img_control -= 5
        else:
            img_control = 0
            print("First img already")
    # 快速进
    elif command == ord('c'):
        if img_control + 4 < img_control_max:
            img_control += 5
        else:
            img_control = img_control_max
            print("Last img already")
    # 退出
    elif command == ord('q'):
        break
    else:
        print("Unknown Command")


