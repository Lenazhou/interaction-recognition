from __future__ import  absolute_import
from __future__ import  division
from data.bit.pose.bit_dataset import BITDataset


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = BITDataset(opt)
        #self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        skeletons, label= self.db.get_example(idx)
        #img_g,person_boxes= self.tsf((img, person_boxes))

        return skeletons.copy(),label.copy()
    def __len__(self):
        return len(self.db)

class TestDataset:
    def __init__(self, opt, split='test_name'):
        self.opt = opt
        self.db = BITDataset(opt ,split=split)
    def __getitem__(self,idx):
        skeletons, label= self.db.get_example(idx)

        return skeletons,label

    def __len__(self):
        return len(self.db)
