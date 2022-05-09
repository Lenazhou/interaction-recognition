# coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from models.resnet50_gate import MyResnetGate
import argparse
import yaml
from models.utils.confusion_matrix import plot_confusion_matrix


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='two-stream Network')
    parser.add_argument('--train', default=True)
    parser.add_argument('--dataset', default='ut')
    parser.add_argument('--config', default='./config/ut/default-a.yaml')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
            default_arg['roi_size'] = tuple(default_arg['roi_size'])
            default_arg['bilinear_size'] = tuple(default_arg['bilinear_size'])
            default_arg['feature_size'] = tuple(default_arg['feature_size'])
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()
    if arg.train=='True':
        arg.train = True
    else:
        arg.train = False
    resnet50 = MyResnetGate(n_class=arg.num_class, roi_size=arg.roi_size, bilinear_size=arg.bilinear_size,
                                 batch_size=arg.batch_size, feature_size=arg.feature_size).cuda()
    if not arg.train:
        pretrained_dict = torch.load(arg.pretrain_model)
        resnet50.load_state_dict(pretrained_dict)
    if arg.dataset == 'bit':
        from data.bit.bit_dataset import Dataset, Test_Dataset
        from models.utils.tav import train_and_valid
        from models.utils.appearance_test import computeTestSetAccuracy
        action = ['BD', 'PC', 'HS', 'HF', 'HG', 'KK', 'PT', 'PS', 'OT']
        loss_func = (nn.CrossEntropyLoss()).cuda()
    elif arg.dataset == 'ut':
        from data.ut.ut_dataset import Dataset,Test_Dataset
        from models.utils.tav import train_and_valid
        from models.utils.appearance_test import  computeTestSetAccuracy
        action = ['HS','HG','KK','PO','PC','PS','OT']
        loss_func = (nn.CrossEntropyLoss()).cuda()
    elif arg.dataset == 'ci':
        from data.ci.ci_dataset import Dataset,Test_Dataset
        from models.utils.tav_ci import train_and_valid
        from models.utils.appearance_test_ci import computeTestSetAccuracy
        action = ['HS','PC','HG','KK','HF','SP','PK','TK','PA','OT']
        class_weight = torch.from_numpy(np.array(
            [6.757625050833673, 28.89913043478261, 9.596881316777361, 50.58447488584475, 14.362143474503027,
             23.772532188841204, 23.80659025787966, 0.7373208501575188, 34.40372670807454, 0.1])).float()
        loss_func = (nn.CrossEntropyLoss(weight=class_weight)).cuda()

    dataset = Dataset(arg)
    train_data = DataLoader(dataset, batch_size=arg.batch_size, shuffle=True, num_workers=arg.num_workers)
    test_dataset = Test_Dataset(arg)
    test_data = DataLoader(test_dataset, batch_size=arg.batch_size, shuffle=False, num_workers=arg.num_workers)
    optimizer = optim.Adam(resnet50.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)

    if arg.train:
        train_and_valid(resnet50, train_data, test_data, loss_func, optimizer, action, arg)
    else:
        cmValue, avg_test_acc, avg_test_loss = computeTestSetAccuracy(resnet50, test_data, loss_func, arg)
        print(avg_test_acc)
        print(avg_test_loss)
        plot_confusion_matrix(cmValue, avg_test_acc, 0, arg, classes=action, normalize=True, title='')