from data.ci.util import random_flip,flip_bbox,random_crop_resize,flip_skeleton,resize_bbox
from data.ci.ci_gendata import CIDataset

def transform_bit(img,bbox,skeleton):
    bbox = resize_bbox(bbox,(1080,1920),(216,384))
    img, params = random_flip(img, x_random=True, return_param=True)
    _,o_H, o_W = img.shape
    bbox = flip_bbox(bbox, (o_H, o_W), x_flip=params['x_flip'])
    skeleton = flip_skeleton(skeleton, x_flip=params['x_flip'])
    img, bbox,skeleton = random_crop_resize(img, bbox,skeleton, crop_random=True)
    return img,bbox,skeleton

class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = CIDataset(opt)
    def __getitem__(self, item):
        img, bbox, label, group_p, group_num, bbox_num,skeleton,label_s,dis_flag = self.db.get_example(item)
        img, bbox,skeleton = transform_bit(img, bbox,skeleton)
        return img.copy(), bbox.copy(), label.copy(), group_p.copy(), group_num.copy(), bbox_num.copy(),skeleton.copy(),label_s.copy(),dis_flag.copy()

    def __len__(self):
        return len(self.db)

class Test_Dataset:
    def __init__(self, opt, split='test_name'):
        self.opt = opt
        self.db = CIDataset(opt, split=split)
    def __getitem__(self, item):
        img, bbox, label, group_p, group_num, bbox_num,skeleton,label_s,dis_flag = self.db.get_example(item)
        bbox = resize_bbox(bbox,(1080,1920),(216,384))
        return img, bbox, label, group_p, group_num, bbox_num,skeleton,label_s,dis_flag

    def __len__(self):
        return len(self.db)
