from __future__ import  absolute_import
from __future__ import  division
from data.ut.pose.ut_dataset import UTDataset

class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = UTDataset(opt)

    def __getitem__(self, idx):
        skeletons, label= self.db.get_example(idx)
        return skeletons.copy(),label.copy()
    def __len__(self):
        return len(self.db)

class TestDataset:
    def __init__(self, opt, split='test_name'):
        self.opt = opt
        self.db = UTDataset(opt, split=split)
    def __getitem__(self,idx):
        skeletons, label= self.db.get_example(idx)
        return skeletons,label

    def __len__(self):
        return len(self.db)
