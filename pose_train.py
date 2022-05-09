#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import time
import numpy as np
import yaml
import pickle
from collections import OrderedDict
from torchnet import meter
import matplotlib.pyplot as plt
import itertools
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
# from tensorboardX import SummaryWriter
import shutil
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import random
import inspect
import torch.backends.cudnn as cudnn

def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Shift Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument('-Experiment_name', default='')
    parser.add_argument('--dataset',default='')
    parser.add_argument(
        '--config',
        default='./config/bit/default-p.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default='',
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument('--only_train_part', default=True)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):

        arg.model_saved_name = "./work_dir/"+arg.Experiment_name+"/weight"
        arg.work_dir = "./work_dir/"+arg.Experiment_name
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)

        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0

    def load_data(self):
        self.data_loader = dict()
        if self.arg.dataset == 'bit':
            from data.bit.pose.dataset import Dataset, TestDataset
            self.action = ['BD', 'PC', 'HS', 'HF', 'HG', 'KK', 'PT','PS','OT']
        elif self.arg.dataset == 'ut':
            from data.ut.pose.datasetUT import Dataset,TestDataset
            self.action = ['HS', 'HG', 'KK', 'PO', 'PC', 'PS', 'OT']
        elif self.arg.dataset == 'ci':
            from data.ci.pose.dataset import Dataset,TestDataset
            self.action = ['HS', 'PC', 'HG', 'KK', 'HF', 'SP', 'PK', 'TK', 'PA', 'OT']
        '''
        q = [i for i in range(1, 51)]
        random.shuffle(q)

        a = random.sample(q, 16)
        b = [i for i in range(1, 51) if i not in a]
        
        a=[9, 37, 5, 17, 8, 32, 29, 31, 42, 25, 14, 7, 45, 2, 41, 28]
        b=[1, 3, 4, 6, 10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 23, 24, 26, 27, 30, 33, 34, 35, 36, 38, 39, 40, 43, 44, 46, 47, 48, 49, 50]
        print(a)
        print(b)
        '''
        if self.arg.phase == 'train':

            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Dataset(self.arg),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=TestDataset(self.arg),
            batch_size=self.arg.batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        if self.arg.dataset == 'ci':
            class_weight = torch.from_numpy(np.array(
            [6.757625050833673, 28.89913043478261, 9.596881316777361, 50.58447488584475, 14.362143474503027,
             23.772532188841204, 23.80659025787966, 0.7373208501575188, 34.40372670807454, 0.1])).float()
            self.loss = nn.CrossEntropyLoss(weight=class_weight).cuda(output_device)
        else:
            self.loss = nn.CrossEntropyLoss().cuda(output_device)
        torch.backends.cudnn.enabled = False
        if self.arg.weights:
            pretrain_model = torch.load(self.arg.weights)
            self.model.load_state_dict(pretrain_model)
        # if self.arg.weights:
        #     # self.global_step = int(arg.weights[:-3].split('-')[-1])
        #     self.print_log('Load weights from {}.'.format(self.arg.weights))
        #     if '.pkl' in self.arg.weights:
        #         with open(self.arg.weights, 'r') as f:
        #             weights = pickle.load(f)
        #     else:
        #         weights = torch.load(self.arg.weights)
        #
        #     weights = OrderedDict(
        #         [[k.split('module.')[-1],
        #           v.cuda(output_device)] for k, v in weights.items()])
        #
        #     for w in self.arg.ignore_weights:
        #         if weights.pop(w, None) is not None:
        #             self.print_log('Sucessfully Remove Weights: {}.'.format(w))
        #         else:
        #             self.print_log('Can Not Remove Weights: {}.'.format(w))
        #
        #     try:
        #         self.model.load_state_dict(weights)
        #     except:
        #         state = self.model.state_dict()
        #         diff = list(set(state.keys()).difference(set(weights.keys())))
        #         print('Can not find these weights:')
        #         for d in diff:
        #             print('  ' + d)
        #         state.update(weights)
        #         self.model.load_state_dict(state)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':

            params_dict = dict(self.model.named_parameters())
            params = []

            for key, value in params_dict.items():
                decay_mult = 0.0 if 'bias' in key else 1.0

                lr_mult = 1.0
                weight_decay = 1e-4
                if 'Linear_weight' in key:
                    weight_decay = 1e-3
                elif 'Mask' in key:
                    weight_decay = 0.0
                    
                params += [{'params': value, 'lr': self.arg.base_lr, 'lr_mult': lr_mult, 'decay_mult': decay_mult, 'weight_decay': weight_decay}]

            self.optimizer = optim.SGD(
                params,
                momentum=0.9,
                nesterov=self.arg.nesterov)
            
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1,
                                              patience=10, verbose=True,
                                              threshold=1e-4, threshold_mode='rel',
                                              cooldown=0)

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
            # os.makedirs(self.arg.work_dir+'/eval_results')
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        loss_value = []
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)

        if epoch >= self.arg.only_train_epoch:
            for key, value in self.model.named_parameters():
                if 'PA' in key:
                    value.requires_grad = True
                    print(key + '-require grad')
        else:
            for key, value in self.model.named_parameters():
                if 'PA' in key:
                    value.requires_grad = False
                    print(key + '-not require grad')
        for batch_idx, (skeletons,labels_s) in enumerate(process):
            if len(labels_s)!=self.arg.batch_size:
                continue
            self.global_step += 1
            real_skeletons = list()
            real_labels = list()
            for b in range(self.arg.batch_size):
                s=skeletons[b]
                l=labels_s[b]
                for j,i in enumerate(l):
                    if i!=-1:
                        real_skeletons.append(s[j])
                        real_labels.append(i)
            real_skeletons = np.stack(real_skeletons).astype(np.float32)
            real_labels = np.stack(real_labels).astype(np.int32)
            real_skeletons = torch.tensor(real_skeletons)
            real_labels = torch.tensor(real_labels)
            # get data
            data = Variable(real_skeletons.float().cuda(self.output_device), requires_grad=False)
            label = Variable(real_labels.long().cuda(self.output_device), requires_grad=False)
            timer['dataloader'] += self.split_time()

            # forward
            start = time.time()
            output = self.model(data)
            network_time = time.time()-start

            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.data)
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']

            if self.global_step % self.arg.log_interval == 0:
                self.print_log(
                    '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}  network_time: {:.4f}'.format(
                        batch_idx, len(loader), loss.data, self.lr, network_time))
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }


    def eval(self, epoch, save_model ,save_score=False, loader_name=['test'],wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        confusion_matrix = meter.ConfusionMeter(self.arg.num_class)  # 存储混淆矩阵的数据
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            right_num_total = 0
            total_num = 0
            loss_total = 0
            step = 0
            process = tqdm(self.data_loader[ln])
            for batch_idx, (skeletons,labels_s) in enumerate(process):
                if len(labels_s) != self.arg.batch_size:
                    continue
                self.global_step += 1
                real_skeletons = list()
                real_labels = list()
                for b in range(self.arg.batch_size):
                    s = skeletons[b]
                    l = labels_s[b]
                    for j, i in enumerate(l):
                        if i != -1:
                            real_skeletons.append(s[j])
                            real_labels.append(i)
                real_skeletons = np.stack(real_skeletons).astype(np.float32)
                real_labels = np.stack(real_labels).astype(np.int32)
                real_skeletons = torch.tensor(real_skeletons)
                real_labels = torch.tensor(real_labels)

                total_num +=len(real_labels)
                data = Variable(
                    real_skeletons.float().cuda(self.output_device),
                    requires_grad=False,
                    volatile=True)
                label = Variable(
                    real_labels.long().cuda(self.output_device),
                    requires_grad=False,
                    volatile=True)

                with torch.no_grad():
                    output = self.model(data)

                loss = self.loss(output, label)
                score_frag.append(output.data.cpu().numpy())
                loss_value.append(loss.data.cpu().numpy())

                _, predict_label = torch.max(output.data, 1)
                acc = torch.sum((predict_label == label.data).float())
                right_num_total +=acc
                step += 1

                confusion_matrix.add(output.squeeze(), label)


            score = np.concatenate(score_frag)

            #accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            accuracy = right_num_total/total_num

            if accuracy > self.best_acc and save_model:
                self.best_acc = accuracy
                print('\tbest accuracy now :'+str(self.best_acc.item()))
                cmValue = confusion_matrix.value()
                self.plot_confusion_matrix(epoch,self.best_acc.item(), cmValue, self.arg, classes=self.action, normalize=True, title='')
                state_dict = self.model.state_dict()
                weights = OrderedDict([[k.split('module.')[-1],
                                        v.cpu()] for k, v in state_dict.items()])

                torch.save(weights,
                           self.arg.model_saved_name + '-' + str(epoch) + '-' + str(self.best_acc.item()) + '.pt')

            self.print_log('Eval Accuracy: '+str(accuracy.item()))


    def train_ci(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        loss_value = []
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)

        if epoch >= self.arg.only_train_epoch:
            for key, value in self.model.named_parameters():
                if 'PA' in key:
                    value.requires_grad = True
                    print(key + '-require grad')
        else:
            for key, value in self.model.named_parameters():
                if 'PA' in key:
                    value.requires_grad = False
                    print(key + '-not require grad')
        right_num_total = 0
        total_num = 0
        #skeletons[32,105,3,1,36,1] labels_s[32,105] count[32,]
        for batch_idx, (skeletons,labels_s,count,dis_flag) in enumerate(process):
            if len(labels_s)!=self.arg.batch_size:
                continue
            self.global_step += 1
            real_skeletons = list()
            real_labels = list()
            ot_labels = list()
            for b in range(self.arg.batch_size):
                b_dis =dis_flag[b][:count[b]]
                b_skeletons = skeletons[b][:count[b]]
                b_labels = labels_s[b][:count[b]]
                for d_id,dis in enumerate(b_dis):
                    if dis ==1:
                        real_skeletons.append(b_skeletons[d_id])
                        real_labels.append(b_labels[d_id])
                    else:
                        ot_labels.append(b_labels[d_id])
            real_skeletons = torch.stack(real_skeletons,0)
            real_labels = torch.Tensor(real_labels)
            # get data
            data = Variable(real_skeletons.float().cuda(self.output_device), requires_grad=False)
            label = Variable(real_labels.long().cuda(self.output_device), requires_grad=False)
            timer['dataloader'] += self.split_time()
            total_num += (len(real_labels) + len(ot_labels))
            # forward
            start = time.time()
            output = self.model(data)
            network_time = time.time()-start

            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.data)
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc_sum = torch.sum((predict_label == label.data).float())
            for ot in ot_labels:
                if ot == 9:
                    acc_sum +=1
            right_num_total +=acc_sum

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            accuracy = (right_num_total/total_num)*100
            if self.global_step % self.arg.log_interval == 0:
                self.print_log(
                    '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}  network_time: {:.4f} accuracy : {:.6f}%'.format(
                        batch_idx, len(loader), loss.data, self.lr, network_time, accuracy))
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }

    def eval_ci(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        confusion_matrix = meter.ConfusionMeter(self.arg.num_class)  # 存储混淆矩阵的数据
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            right_num_total = 0
            total_num = 0
            loss_total = 0
            step = 0
            process = tqdm(self.data_loader[ln])
            for batch_idx, (skeletons, labels_s, count, dis_flag) in enumerate(process):
                if len(labels_s) != self.arg.batch_size:
                    continue
                self.global_step += 1
                real_skeletons = list()
                real_labels = list()
                ot_labels = list()
                for b in range(self.arg.batch_size):
                    b_dis = dis_flag[b][:count[b]]
                    b_skeletons = skeletons[b][:count[b]]
                    b_labels = labels_s[b][:count[b]]
                    for d_id, dis in enumerate(b_dis):
                        if dis == 1:
                            real_skeletons.append(b_skeletons[d_id])
                            real_labels.append(b_labels[d_id])
                        else:
                            ot_labels.append(b_labels[d_id])
                real_skeletons = torch.stack(real_skeletons, 0)
                real_labels = torch.Tensor(real_labels)

                total_num += (len(real_labels) + len(ot_labels))
                data = Variable(
                    real_skeletons.float().cuda(self.output_device),
                    requires_grad=False,
                    volatile=True)
                label = Variable(
                    real_labels.long().cuda(self.output_device),
                    requires_grad=False,
                    volatile=True)

                with torch.no_grad():
                    output = self.model(data)

                loss = self.loss(output, label)
                score_frag.append(output.data.cpu().numpy())
                loss_value.append(loss.data.cpu().numpy())

                _, predict_label = torch.max(output.data, 1)
                acc = torch.sum((predict_label == label.data).float())

                for ot in ot_labels:
                    if ot == 9:
                        acc += 1
                right_num_total += acc
                step += 1

                confusion_matrix.add(output.squeeze(), label)

            score = np.concatenate(score_frag)

            # accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            accuracy = right_num_total / total_num

            if accuracy > self.best_acc:
                self.best_acc = accuracy

                state_dict = self.model.state_dict()
                weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
                torch.save(weights,
                           self.arg.model_saved_name + '-' + str(epoch) + '-' + str(self.best_acc.item()) + '.pt')

                print('\tbest accuracy now :' + str(self.best_acc.item()))
                cmValue = confusion_matrix.value()

                self.plot_confusion_matrix(epoch, self.best_acc.item(), cmValue, self.arg, classes=self.action, normalize=True,
                                           title='')

            self.print_log('Eval Accuracy: ' + str(accuracy.item()))

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                if self.arg.dataset == 'ci':
                    self.train_ci(epoch)
                    self.eval_ci(epoch, save_model= True, save_score=self.arg.save_score,loader_name=['test'])
                else:
                    self.train(epoch)
                    self.eval(
                        epoch,
                        save_model = True,
                        save_score=self.arg.save_score,
                        loader_name=['test'])

            print('best accuracy: ', self.best_acc, ' model_name: ', self.arg.model_saved_name)

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = self.arg.model_saved_name + '_wrong.txt'
                rf = self.arg.model_saved_name + '_right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')

    def plot_confusion_matrix(self, epoch, accuracy, cm, arg,classes, normalize=False, title='Confusion matrix',
                              cmap="BuPu"):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        Input
        - cm : 计算出的混淆矩阵的值
        - classes : 混淆矩阵中每一行每一列对应的列
        - normalize : True:显示百分比, False:显示个数
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        # print(cm)

        f = plt.figure(figsize=(7, 7))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        # plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.tick_params(labelsize=16)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "red",
                     size=15)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        path = os.path.join(arg.work_dir, 'visualization', 'epoch-' + str(epoch) + '-acc-' + str(accuracy) + '.jpg')
        f.savefig(path)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        # for k in default_arg.keys():
        #     if k not in key:
        #         print('WRONG ARG: {}'.format(k))
        #         assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(0)
    processor = Processor(arg)
    processor.start()
