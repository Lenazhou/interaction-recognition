import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from data.bit.util import get_box_tensor
from torchvision.ops import RoIAlign
import time

class MyResNet50(nn.Module):
    def __init__(self,pretrained=False):
        super(MyResNet50,self).__init__()
        resnet=models.resnet50(pretrained=pretrained)
        resnet.load_state_dict(torch.load('/home/zhenghua/zj/resnet50/interaction/ut_set/resnet50-19c8e357.pth'))
        self.conv1=resnet.conv1
        self.bn1=resnet.bn1
        self.relu=resnet.relu
        self.maxpool=resnet.maxpool

        self.layer1=resnet.layer1
        self.layer2=resnet.layer2
        self.layer3=resnet.layer3
        self.layer4=resnet.layer4

    def forward(self, x):
        outputs=[]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)#(N, 64, 56, 56)
        #outputs.append(x1)
        x2 = self.layer1(x1)#(N, 256, 56, 56)
        #outputs.append(x2)
        x3 = self.layer2(x2)#(N, 512, 28, 28)
        #outputs.append(x3)
        x4 = self.layer3(x3)#(N, 1024, 14, 14)
        outputs.append(x4)
        x5 = self.layer4(x4)#(N, 2048, 7, 7)
        outputs.append(x5)

        return outputs


#https://github.com/guopei/PoseEstimation-FCN-Pytorch
class MyResnetGate(nn.Module):
    def __init__(self, n_class,roi_size, bilinear_size,batch_size,feature_size):
        super(MyResnetGate, self).__init__()

        self.features_base = MyResNet50()
        self.bilinear_size = bilinear_size
        self.feature_size = feature_size
        self.n_class = n_class
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(1,1,1)
        self.sigmod = nn.Sigmoid()
        self.roialign = RoIAlign(roi_size, spatial_scale=1, sampling_ratio=1)
        self.norm1 = nn.LayerNorm(38400)
        self.norm2 = nn.LayerNorm(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.9)
        self.fc1 = nn.Linear(38400,512)
        self.fc2 = nn.Linear(512,n_class)
        self.conv2 = nn.Conv2d(1024,512,1)
        self.conv3 = nn.Conv2d(2048,1024,1)
        # 初始化线性层
        nn.init.kaiming_normal_(self.fc1.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    #x[B,3,240,320]
    def forward(self, inputs, boxes, group,real_in_num,label_s,skeleton):

        x = self.features_base(inputs)
        resnet_f = list() #包含5层的特征

        for index,x_i in enumerate(x):
            #使用双线性插值将不同的维度的信息转化（240,320）
            if index==0:
                x_i = self.conv2(x_i)
            if index==1:
                x_i = self.conv3(x_i)
            x_i = F.interpolate(x_i, size=self.feature_size, mode='bilinear', align_corners=True)
            resnet_f.append(x_i)

        x = torch.cat(resnet_f, dim=1) #（B,3904,240,320）

        sk_f=list()

        for bt in range(self.batch_size):
            app_feature = x[bt]
            poses = skeleton[bt]
            for j,g in enumerate(group[bt][:real_in_num[bt]]):
                if label_s[bt][j]!=-1:
                    pose = poses[j]
                    pose = pose[None,None,:,:]
                    pose = self.conv1(pose)
                    pose = self.sigmod(pose)
                    pose_feature = pose*app_feature
                else:
                    pose_feature = app_feature
                    pose_feature = pose_feature[None,:,:,:]
                b1 = boxes[bt][g[0]]
                b2 = boxes[bt][g[1]]
                box = get_box_tensor(b1,b2)
                box = box[None,:]
                box = [box.cuda()]
                roi = self.roialign(pose_feature,box)
                sk_f.append(roi)

        sk_f = torch.cat(sk_f,dim=0)
        sk_f=sk_f.view(-1,38400)
        sk_f = self.norm1(sk_f)
        sk_f = self.relu(sk_f)
        sk_f = self.dropout(sk_f)
        sk_f = self.fc1(sk_f)


        sk_f = self.norm2(sk_f)
        score = self.fc2(sk_f)

        return score










