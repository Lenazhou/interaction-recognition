#coding = utf-8
import os
from .util import load_json
import numpy as np

class UTDataset:
    def __init__(self, opt, split='train_name'):
        id_list_file = os.path.join(
            opt.data_dir,'data', opt.dataset, '{0}.npy'.format(split))
        self.ids = np.load(id_list_file)
        self.data_dir = opt.data_dir
        self.dataset_dir = opt.dataset_dir
        self.opt = opt

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        id_ = self.ids[i]  # 'push/push_0047/24.json'
        seq_num = id_.split('-')[0]
        frame_num = (id_.split('-')[1]).replace('.json', '')
        frame_file = os.path.join(self.dataset_dir, 'all2', seq_num, id_)

        info = load_json(frame_file)
        label = list()
        skeletons = np.zeros((self.opt.Max_group_num, 3, 1, 36, 1), dtype=np.float32)
        count=0
        for k,v in info['interact'].items():
            group = v['group']
            if info['bodyinfo'][group[0]]['coor'] and info['bodyinfo'][group[1]]['coor']:
                coorlist = list()
                coorlist += info['bodyinfo'][group[0]]['coor']
                coorlist += info['bodyinfo'][group[1]]['coor']
                data = read_xyz(coorlist)
                label.append(BIT_BBOX_LABEL_NAMES.index(v['action']))
                skeletons[count,:,:,:,:]=data
                count+=1
            else:
                label.append(-1)
                count+=1

        while len(label)!=self.opt.Max_group_num:
            label.append(-1)

        # if opt.bone:
        #     skeletons = gen_bone(skeletons)

        skeletons = np.stack(skeletons).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        return skeletons, label

    __getitem__ = get_example

BIT_BBOX_LABEL_NAMES = (
            'hs',
            'hg',
            'kk',
            'pt',
            'pc',
            'ps',
            'na'
        )

def read_xyz(coorlist):
    # one file one group one frame 34 joints 3 channels
    data = np.zeros((1, 1, 1, 36, 3), dtype=np.float32)  # N M T V C
    for k in range(1):
        for i in range(1):
            for j in range(1):
                for n, joint in enumerate(coorlist):
                    data[k, i, j, n, :] = [joint[0], joint[1], 0]
    data = data.transpose(0, 4, 2, 3, 1)  # 转置
    return data

def gen_bone(skeletons,opt):
    paris = {
        'ut_interact': (
            (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6),
            (8, 1), (9, 8), (10, 9), (11, 1), (12, 11), (13, 12),
            (14, 0), (15, 0), (16, 14), (17, 15), (19, 18), (20, 19),
            (21, 20), (22, 21), (23, 19), (24, 23), (25, 24),
            (26, 19), (27, 26), (28, 27), (29, 19), (30, 29),
            (31, 30), (32, 18), (33, 18), (34, 32), (35, 33)
        )
    }
    bones = np.zeros((opt.Max_in_n, 3, 1, 36, 1), dtype=np.float32)
    for v1, v2 in paris['ut_interact']:
        bones[:, :, :, v1, :] = skeletons[:, :, :, v1, :] - skeletons[:, :, :, v2, :]

    return bones