#coding = utf-8
import os
import json
import numpy as np
import pickle
from PIL import Image
import cv2
import time
from data.bit.util import get_files_id
class BITDataset:

    def __init__(self, opt, split='train_name'):
        '''
        id_list_file = os.path.join(
            data_dir, 'data/bit/{0}.pkl'.format(split))

        with open(id_list_file, 'rb') as f:
            self.ids= pickle.load(f)
        '''
        self.ids = get_files_id(split)
        self.data_dir = opt.data_dir
        self.dataset_dir = opt.dataset_dir
        self.opt = opt
    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        id_ = self.ids[i]
        action = id_.split('/')[0]
        action_num = id_.split('/')[1]
        frame_file = os.path.join(self.dataset_dir,'all',id_)
        info = load_json(frame_file)
        id_2=id_.replace('.json','.npy')

        skeleton = np.load(os.path.join(self.dataset_dir,'pose2',id_2))

        #cnn bbox label_cnn label_cnn_num bbox_num image group
        bbox = list()
        label = list()
        group_p = list()

        for j,bx in info['bodyinfo'].items():
            try:
                if bx['bbox']:
                    bbox.append(bx['bbox'])
                else:
                    bbox.append([0., 0., 0., 0.])
            except Exception as e:
                print(e)
                print(id_)
                continue


        bbox_num = len(bbox)
        label_s=list()
        for k,v in info['interact'].items():
            group = v['group']
            if info['bodyinfo'][group[0]]['bbox'] and info['bodyinfo'][group[1]]['bbox']:
                g = list(map(int, group))
                group_p.append(g)
                label.append(BIT_BBOX_LABEL_NAMES.index(v['action']))
                if info['bodyinfo'][group[0]]['box'] and info['bodyinfo'][group[1]]['box']:
                    label_s.append(BIT_BBOX_LABEL_NAMES.index(v['action']))
                else:
                    label_s.append(-1)

        group_num = len(label)

        while len(label) != self.opt.max_group_num:
            label.append(-1)
            label_s.append(-1)
            group_p.append(([-1,-1]))

        while len(bbox)!= self.opt.max_box_num:
            bbox.append([0.,0.,0.,0.])

        img=read_img(self.opt, action,action_num,id_)

        bbox = np.stack(bbox).astype(np.float32)

        label = np.stack(label).astype(np.int32)
        label_s = np.stack(label_s).astype(np.int32)
        group_p = np.stack(group_p).astype(np.int32)
        group_num = np.array(group_num,dtype=np.int32)
        bbox_num = np.array(bbox_num,dtype=np.int32)
        img = np.array(img,dtype=np.float32)
        skeleton = np.array(skeleton,dtype=np.float32)

        return img, bbox, label, group_p, group_num, bbox_num,skeleton,label_s

    __getitem__ = get_example

BIT_BBOX_LABEL_NAMES = (
            'bend',
            'box',
            'handshake',
            'highfive',
            'hug',
            'kick',
            'pat',
            'push',
            'no_action'
        )

def read_img(opt, action,action_num,id_):
    img_num = (((id_.split('/')[-1]).replace('.json','')).zfill(4))+'.jpg'
    path = os.path.join(opt.dataset_dir,'Bit-frames',action,action_num,img_num)
    f = Image.open(path)
    img = f.convert('RGB')
    img = np.asarray(img)
    img = img.transpose((2,0,1))
    return img

def load_json(p):
	with open(p,'r') as f:
		load_dict=json.load(f)
	return load_dict