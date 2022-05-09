from __future__ import  absolute_import
from __future__ import  division

from data.ci.pose.ci_dataset import CIDataset


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = CIDataset(opt)
    def __getitem__(self, idx):
        skeletons, label, count, dis_flag= self.db.get_example(idx)
        return skeletons.copy(),label.copy(),count.copy(),dis_flag.copy()
    def __len__(self):
        return len(self.db)

class TestDataset:
    def __init__(self, opt, split='test_name'):
        self.opt = opt
        self.db = CIDataset(opt, split=split)
    def __getitem__(self,idx):
        skeletons, label, count,dis_flag= self.db.get_example(idx)

        return skeletons,label,count,dis_flag

    def __len__(self):
        return len(self.db)
