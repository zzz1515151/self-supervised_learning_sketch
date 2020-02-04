import os 
import numpy as np 
import csv

def label2idx(labels):
    label2idxs = {}
    for idx, label in enumerate(labels):
        if not label in label2idxs:
            label2idxs[label] = idx    
    return label2idxs

def get_file_list(dataset_root):
    '''get classes'''
    assert os.path.exists(dataset_root), "{} does not exist".format(dataset_root)
    labels = []    
    for class_name in os.listdir(dataset_root):
        class_path = os.path.join(dataset_root, class_name)
        if os.path.isdir(class_path):
            labels.append(class_name)            
    """class to idx"""
    label2idxs = label2idx(labels)
    image_info = []
    for class_name in labels:
        class_path = os.path.join(dataset_root, class_name)
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            image_info.append([file_path, label2idxs[class_name]])
    return image_info, label2idxs                             



if __name__ == '__main__':                                                     

    dataset_root = '../datasets/quickdraw_224png_20190108'
    image_csv_dir = '../datasets/quickdraw_224png_20190108.csv'
    class_dir = '../datasets/quickdraw_224png_20190108_classes.csv'
    print("Begin generating file from {}".format(dataset_root))
    print("Saving image list file {}".format(image_csv_dir))
    print("Saving class list file {}".format(class_dir))
    image_info, label2idxs = get_file_list(dataset_root)
    image_csv_file = csv.writer(open(image_csv_dir,'a',newline=''),dialect='excel')
    class_csv_file = csv.writer(open(class_dir, 'a', newline = ''), dialect = 'excel')
    for info in image_info:
        image_csv_file.writerow(info)
    assert len(label2idxs.keys()) == 345
    for item in label2idxs.keys():
        class_csv_file.writerow([item, label2idxs[item]])
    print("File list generated")

