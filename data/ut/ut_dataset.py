from data.ut.util import random_flip,flip_bbox,flip_skeleton,resize_bbox,random_crop_resize_ut
from data.ut.ut_gendata import UTDataset
import time

def transform_bit(img,bbox,skeleton):
    bbox = resize_bbox(bbox,(480,720),(240,360)) #(240,360)
    img, params = random_flip(img, x_random=True, return_param=True)
    _,o_H, o_W = img.shape
    bbox = flip_bbox(bbox, (o_H, o_W), x_flip=params['x_flip'])
    skeleton = flip_skeleton(skeleton, x_flip=params['x_flip'])
    img, bbox,skeleton = random_crop_resize_ut(img, bbox,skeleton, crop_random=True)
    return img,bbox,skeleton

class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = UTDataset(opt)
    def __getitem__(self, item):
        img, bbox, label, group_p, group_num, bbox_num,skeleton,label_s = self.db.get_example(item)
        img, bbox,skeleton = transform_bit(img, bbox,skeleton)

        return img.copy(), bbox.copy(), label.copy(), group_p.copy(), group_num.copy(), bbox_num.copy(),skeleton.copy(),label_s.copy()

    def __len__(self):
        return len(self.db)

class Test_Dataset:
    def __init__(self,opt, split='test_name'):
        self.opt = opt
        self.db = UTDataset(opt, split=split)
    def __getitem__(self, item):
        img, bbox, label, group_p, group_num, bbox_num,skeleton,label_s = self.db.get_example(item)
        bbox = resize_bbox(bbox, (480, 720), (240, 360))
        return img, bbox, label, group_p, group_num, bbox_num,skeleton,label_s

    def __len__(self):
        return len(self.db)
