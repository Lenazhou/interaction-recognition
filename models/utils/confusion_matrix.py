import matplotlib.pyplot as plt
import numpy as np
import itertools
import os

# 绘制混淆矩阵
def plot_confusion_matrix(cm, avg_acc, epoch, arg, classes, normalize=False, title='Confusion matrix', cmap="BuPu"):
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
    path = os.path.join(arg.work_dir, 'appearance/visualization', str(epoch+1)+ '_'+str(avg_acc)+'.jpg')
    f.savefig(path)
