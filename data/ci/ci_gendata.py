#coding = utf-8
import os
import json
import numpy as np
import pickle
from PIL import Image
import cv2
import time
from skimage import transform as sktsf
from data.ci.util import cal_box_center
class CIDataset:

    def __init__(self, opt, split='train_name'):
        id_list_file = os.path.join(
            opt.data_dir, 'data/ci/{0}.pkl'.format(split)) #读取每张图片的name:IMG_1127/001

        with open(id_list_file, 'rb') as f:
            self.ids= pickle.load(f)
        self.data_dir = opt.data_dir  #当前项目的名字
        self.dataset_dir = opt.dataset_dir #数据集的根目录
        self.pose_dir = opt.pose_dir #骨架注意力图的根目录
        self.opt = opt

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        id_ = self.ids[i] #IMG_1127/001
        video = id_.split('/')[0]
        frame_num = (id_.split('/')[1]).replace('.json','') #ann地址
        frame_file = os.path.join(self.dataset_dir,'all',id_) #图片地址
        info = load_json(frame_file)  #读取ann

        id_2=id_.replace('.json','.npy')

        skeleton = np.load(os.path.join(self.pose_dir,id_2))

        #cnn bbox label_cnn label_cnn_num bbox_num image group
        bbox = list()
        label = list()
        group_p = list()

        for j,bx in info['bodyinfo'].items():
            if bx['bbox']:
                bbox.append(bx['bbox']) #获取每个人的bounding box
            else:
                bbox.append([0.,0.,0.,0.]) #bounding box 缺失时为0进行补充

        bbox_num = len(bbox)
        label_s=list()
        dis_flag = list()

        #读取每对交互组的信息
        for k,v in info['interact'].items():
            group = v['group'] #当前交互组中两个人的对应上述bounding box 的id
            #如果当前交互组中两个人都有bounding box才读入，进行判别，缺失bounding box算入缺失问题中
            if info['bodyinfo'][group[0]]['bbox'] and info['bodyinfo'][group[1]]['bbox']:
                g = list(map(int, group))
                group_p.append(g)
                label.append(BIT_BBOX_LABEL_NAMES.index(v['action'])) #获取当前交互组的动作标签
                if info['bodyinfo'][group[0]]['box'] and info['bodyinfo'][group[1]]['box']:
                    label_s.append(BIT_BBOX_LABEL_NAMES.index(v['action'])) #交互组的动作标签转换为数字
                else:
                    label_s.append(-1)

                #计算交互组的两个bounding box的中点点距离，如果超过414，直接判定为ot
                box1 = info['bodyinfo'][group[0]]['bbox']
                box2 = info['bodyinfo'][group[1]]['bbox']
                dis = cal_box_center(box1, box2)
                if dis < 414:
                    dis_flag.append(1)
                else:
                    dis_flag.append(0)

        group_num = len(label)

        #补齐操作，单人数量需要补齐到15，交互组需要补齐到105组
        while len(label)!=self.opt.max_group_num:
            label.append(-1)
            label_s.append(-1)
            dis_flag.append(-1)
            group_p.append(([-1,-1]))

        while len(bbox)!=self.opt.max_box_num:
            bbox.append([0.,0.,0.,0.])

        img=read_img(id_,self.opt)

        bbox = np.stack(bbox).astype(np.float32) #[]每个人

        label = np.stack(label).astype(np.int32) #图片中每一对的交互标签
        label_s = np.stack(label_s).astype(np.int32)
        dis_flag = np.stack(dis_flag).astype(np.int32)
        group_p = np.stack(group_p).astype(np.int32)
        group_num = np.array(group_num,dtype=np.int32)
        bbox_num = np.array(bbox_num,dtype=np.int32)
        img = np.array(img,dtype=np.float32)
        skeleton = np.array(skeleton,dtype=np.float32)

        return img, bbox, label, group_p, group_num, bbox_num,skeleton,label_s,dis_flag

    __getitem__ = get_example

BIT_BBOX_LABEL_NAMES = ('HS','BX','HG','KK','HF','SP','PK','TK','PS','OT')

def read_img(id_,opt):
    id = id_.replace('.json','.jpg')
    path = os.path.join(opt.dataset_dir,'frame',id)
    f = Image.open(path)
    img = f.convert('RGB')
    img = np.asarray(img)
    img = img.transpose((2,0,1))
    img = sktsf.resize(img, (3, 216, 384), mode='reflect', anti_aliasing=False)
    return img


def load_json(p):
	with open(p,'r') as f:
		load_dict=json.load(f)
	return load_dict