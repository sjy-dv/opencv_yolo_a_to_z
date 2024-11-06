from pathlib import Path
import xml.etree.ElementTree as ET
from shutil import copyfile
import os
import numpy as np
import pandas as pd




classes = ['helmet', 'head', 'person']


def convert_annotation(size, box):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    dw = np.float32(1. / int(size[0]))
    dh = np.float32(1. / int(size[1]))

    w = x2 - x1
    h = y2 - y1
    x = x1 + (w / 2)
    y = y1 + (h / 2)

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]



def save_txt_file(img_jpg_file_name, size, img_box):
    save_file_name = './dataset/labels/' +  img_jpg_file_name + '.txt'
    
    #file_path = open(save_file_name, "a+")
    with open(save_file_name ,'a+') as file_path:
        for box in img_box:

            cls_num = classes.index(box[0])

            new_box = convert_annotation(size, box[1:])

            file_path.write(f"{cls_num} {new_box[0]} {new_box[1]} {new_box[2]} {new_box[3]}\n")

        file_path.flush()
        file_path.close()
        
def get_xml_data(file_path, img_xml_file):
    img_path = file_path + '/' + img_xml_file + '.xml'

    tree = ET.parse(img_path)
    root = tree.getroot()

    img_name = root.find("filename").text
    img_size = root.find("size")
    img_w = int(img_size.find("width").text)
    img_h = int(img_size.find("height").text)
    img_c = int(img_size.find("depth").text)

    img_box = []
    for box in root.findall("object"):
        cls_name = box.find("name").text
        x1 = int(box.find("bndbox").find("xmin").text)
        y1 = int(box.find("bndbox").find("ymin").text)
        x2 = int(box.find("bndbox").find("xmax").text)
        y2 = int(box.find("bndbox").find("ymax").text)

        img_box.append([cls_name, x1, y1, x2, y2])

    img_jpg_file_name = img_xml_file + '.jpg'
    save_txt_file(img_xml_file, [img_w, img_h], img_box)
    
    
from tqdm import tqdm

files = os.listdir('./kaggle/helmet/annotations')

for file in tqdm(files, total=len(files)):
    file_xml = file.split(".")
    get_xml_data('./kaggle/helmet/annotations', file_xml[0])
    
# data split
from sklearn.model_selection import train_test_split

# image_list = os.listdir('./kaggle/helmet/annotations')
image_list = [os.path.splitext(file)[0] for file in os.listdir('./kaggle/helmet/annotations')]

# train 80%, 20% test&val
train_list, test_list = train_test_split(image_list, test_size=0.2, random_state=42)

val_list, test_list = train_test_split(test_list, test_size=0.5,random_state=42)

print('total size => ', len(image_list))
print('train size => ', len(train_list))
print('val size => ', len(val_list))
print('test size => ', len(test_list))


# rewrite split train data

def copy_data(file_list, img_labels_root, imgs_source, mode):
    dataset_root = Path('./dataset/')

    # Create directories if they don't exist
    images_path = dataset_root / 'images' / mode
    labels_path = dataset_root / 'labels' / mode
    images_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)

    # Copying files with progress bar
    for file in tqdm(file_list, desc=f"Copying {mode} data"):
        base_filename = file.replace('.png', '')

        img_src_file = Path(imgs_source) / (base_filename + '.png')
        label_src_file = Path(img_labels_root) / (base_filename + '.txt')

        img_dest_file = images_path / (base_filename + '.png')
        label_dest_file = labels_path / (base_filename + '.txt')

        copyfile(img_src_file, img_dest_file)
        copyfile(label_src_file, label_dest_file)

label = './dataset/labels'
dest_img_source = './kaggle/helmet/images'
        
copy_data(train_list, label, dest_img_source, 'train')
copy_data(val_list, label, dest_img_source, 'val')
copy_data(test_list, label, dest_img_source, 'test')