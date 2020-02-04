import torch
import torch.utils.data as data
import torchvision
import torchnet as tnt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import random
from torch.utils.data.dataloader import default_collate
from coordinate_transform_utils import coordinate_transform_utils
from PIL import Image
import os
import errno
import numpy as np
import sys
import csv

def _get_filenames_and_classes(dataset_dir):
    """Returns a list of filenames and inferred class names.

    Args:
      dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
      A list of image file paths, relative to `dataset_dir` and the list of
      subdirectories, representing class names.
    """
    quickdraw_root = dataset_dir
    directories = []
    class_names = []
    for filename in os.listdir(quickdraw_root):
        path = os.path.join(quickdraw_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)

    return photo_filenames, sorted(class_names)

class RNNDataset(data.Dataset):

    def __init__(self, root, split):
        """
        args:
            root:数据集路径
            split:训练集/测试集
        return:
            a pytorch dataset instance 
        """
        self.root = root
        self.split = split.lower()              
        assert(self.split == "train" or self.split == "test")
        _NUM_VALIDATION = 345000
        _RANDOM_SEED = 0  
        photo_filenames, _ = _get_filenames_and_classes(self.root)
        random.seed(_RANDOM_SEED)
        random.shuffle(photo_filenames)
        assert(len(photo_filenames) == 345 * 10000)
        if self.split == "train":
            self.image_list = photo_filenames[_NUM_VALIDATION:]                     
        elif self.split == "test":
            self.image_list = photo_filenames[:_NUM_VALIDATION]        


    def __getitem__(self, index):
        stroke = np.load(self.image_list[index], encoding = "bytes")
        if stroke.shape[0] == 2:
            return stroke[0] 
        else:            
            return stroke 

    def __len__(self):
        return len(self.image_list)

def rotate_rnn(stroke, rot, size):
    w = size[0] -1
    h = size[1] -1
    if rot == 0:
        stroke_0 = [[point[0] / h, point[1]/ w, point[2], point[3]] for point in stroke\
                    if not point[2] == point[3]]
        if not len(stroke_0) == 100:            
            stroke_0 += [[0., 0., 0., 0.] for i in range(100 - len(stroke_0))]
        return np.array(stroke_0).astype(np.float32)
    elif rot == 90:
        stroke_90 = [[(h - point[1]) / h, point[0] / w, point[2], point[3]] for point in stroke \
                     if not point[2] == point[3]]     
        if not len(stroke_90) == 100:   
            stroke_90 += [[0., 0., 0., 0.] for i in range(100 - len(stroke_90))]
        return np.array(stroke_90).astype(np.float32)
    elif rot == 180:
        stroke_180 = [[(w - point[0]) / h, (h - point[1]) / w, point[2], point[3]] for point in stroke \
                      if not point[2] == point[3]]
        if not len(stroke_180) == 100:
            stroke_180 += [[0., 0., 0., 0.] for i in range(100 - len(stroke_180))]
        return np.array(stroke_180).astype(np.float32)
    elif rot == 270:
        stroke_270 = [[point[1] / h, (w - point[0]) / w, point[2], point[3]] for point in stroke \
                      if not point[2] == point[3]]
        if not len(stroke_270) == 100:
            stroke_270 += [[0., 0., 0., 0.] for i in range(100 - len(stroke_270))]
        return np.array(stroke_270).astype(np.float32)
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

def transform_rnn(stroke, a_x = 0., b_x = 0.5, c_x = 1.0, a_y = 0., b_y = 0.5, c_y = 1.0):
    stroke = np.array([[point[0] , point[1], point[2], point[3]] for point in stroke\
                    if not point[2] == point[3]])
    stroke = coordinate_transform_utils(stroke, a_x, b_x, c_x, a_y, b_y, c_y)
    if not stroke.shape[0] == 100:
        stroke_rest = np.array([[0., 0., 0., 0.] for i in range(100 - stroke.shape[0])])
        stroke = np.vstack((stroke, stroke_rest))
    return stroke.astype(np.float32)


class DataLoader(object):
    """
    args:
      dataset: a pytorch dataset instance     
    output:
      a pytorch dataloader instance   
    """
    def __init__(self,
                 dataset,
                 signal_type,
                 batch_size=1,                 
                 epoch_size=None,
                 num_workers=1,
                 shuffle=True,
                 size = (256,256)):
        self.dataset = dataset
        self.signal_type = signal_type
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size        
        self.num_workers = num_workers
        self.size = size
        self.transform = transforms.Compose([
               lambda x:torch.FloatTensor(x) #numpy to tensor                       
              ]) 

    def get_iterator(self, epoch=0):
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)      
        if self.signal_type == 'rotation':
            #无监督，四个方向旋转四分类
            def _load_function(idx):                    
                idx = idx % len(self.dataset)
                stroke = self.dataset[idx]
                rotated_strokes = [
                # 原图
                self.transform(rotate_rnn(stroke, 0, self.size)),
                # 90
                self.transform(rotate_rnn(stroke, 90, self.size)),
                # 180
                self.transform(rotate_rnn(stroke, 180, self.size)),
                # 270
                self.transform(rotate_rnn(stroke, 270, self.size)),
                ]
                rotation_labels=torch.LongTensor([0,1,2,3]) #labels 
                return torch.stack(rotated_strokes, dim=0), rotation_labels
        elif self.signal_type == 'deformation':
            def _load_function(idx):                    
                idx = idx % len(self.dataset)
                stroke = self.dataset[idx]
                rotated_strokes = [
                #原图
                self.transform(rotate_rnn(stroke, 0, self.size)),
                #水平扩展垂直压缩
                self.transform(transform_rnn(stroke, 2.1, 0, 1, -4.1, 0, 1))
                ]
                rotation_labels=torch.LongTensor([0,1]) #labels 
                return torch.stack(rotated_strokes, dim=0), rotation_labels
        else:
            raise ValueError('signal must be rotation or deformation')
        def _collate_fun(batch):
            batch = default_collate(batch)
            assert(len(batch)==2)
            batch_size, rotations, time_step, input_dim = batch[0].size()
            #batch_size * 100 * 4
            batch[0] = batch[0].view([batch_size*rotations, time_step, input_dim])
            batch[1] = batch[1].view([batch_size*rotations])
            return batch

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),\
                                          load=_load_function)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size, \
                  collate_fn=_collate_fun, num_workers=self.num_workers,\
                  shuffle=self.shuffle)
        
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size // self.batch_size
