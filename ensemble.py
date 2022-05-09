import pandas as pd
import numpy as np
import torch
from torchnet import meter
import matplotlib.pyplot as plt
import itertools
from data.bit.pose.util import get_files_id

def plot_confusion_matrix(cm,classes, normalize=False, title='Confusion matrix', cmap="BuPu"):
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
    #print(cm)

    f = plt.figure(figsize=(7, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tick_params(labelsize=15)
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
    f.savefig('3-streams'+'.jpg')

test_name = get_files_id('test_name')
action = ['BD', 'BX', 'HS', 'HF', 'HG', 'KK', 'PT','PS','OT']
confusion_matrix = meter.ConfusionMeter(9)
data1=pd.read_pickle('bit-a.pkl')
data2=pd.read_pickle('bit-p.pkl')

score_two = data1[0]
score_opt = data2[0]
labels = data1[1]
acc_num = 0
for i,j,l in zip(score_two,score_opt,labels):
    output = i+j
    output = torch.from_numpy(output)
    label = torch.from_numpy(np.array([l]))
    confusion_matrix.add(output.unsqueeze(0), label.long())

    if output.argmax()==l:
        acc_num += 1
cmValue=confusion_matrix.value()
plot_confusion_matrix(cmValue, classes=action, normalize=True, title='')
print(acc_num)