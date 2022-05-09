import torch
import os
import time
from torch.autograd import Variable
from models.utils.appearance_test import computeTestSetAccuracy
from models.utils.confusion_matrix import plot_confusion_matrix
# 训练
def train_and_valid(model, train_data, test_data, loss_function, optimizer, action, arg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    w_path = os.path.join(arg.data_dir, 'work_dir', arg.dataset, 'appearance/log/log.txt')
    w_log = open(w_path, 'a')
    best_acc = 0.0
    for epoch in range(arg.epoch):
        epoch_start = time.ctime(time.time())
        print(epoch_start)
        w_log.write(epoch_start + '\n')
        print("Epoch: {}/{}".format(epoch + 1, arg.epoch))

        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_num = 0
        # inputs[B,3,240,320] boxes[B,5,4] labels[B,10] group[B,10,2] real_in_num[B,] real_box_num[B,] skeleton[B,5,18,2] label_s[B,10]
        # real_in_num is the group has box  // real_box_num is the image have box num  // label_s is the group has two skeletons
        for ii, (inputs, boxes, labels, group, real_in_num, real_box_num, skeleton, label_s) in enumerate(train_data):
            if len(inputs) != arg.batch_size:
                continue

            labels_real = list()
            for b in range(arg.batch_size):
                labels_real.append(labels[b, :real_in_num[b]])
            labels_real = torch.cat(labels_real, dim=0)

            inputs = Variable(inputs.float().cuda(device), requires_grad=False)
            boxes = Variable(boxes.float().cuda(device), requires_grad=False)
            labels_real = Variable(labels_real.int().cuda(device), requires_grad=False)
            label_s = Variable(label_s.int().cuda(device), requires_grad=False)
            group = Variable(group.int().cuda(device), requires_grad=False)
            real_in_num = Variable(real_in_num.int().cuda(), requires_grad=False)
            real_box_num = Variable(real_box_num.int().cuda(), requires_grad=False)
            skeleton = Variable(skeleton.float().cuda(), requires_grad=False)
            optimizer.zero_grad()
            train_num += len(labels_real)

            outputs = model(inputs, boxes, group, real_in_num, label_s, skeleton)  # (train_num,7)
            loss = loss_function(outputs, labels_real.long())
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels_real.data.view_as(predictions))
            acc = torch.sum(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item()
            if ii % 50 == 0 and ii != 0:
                avg_acc = train_acc / train_num
                avg_loss = train_loss / train_num
                print(
                    "Trian dataloader ii : {}, avg_acc : {:.4f}%, avg_loss: {:.4f}".format(ii, avg_acc * 100, avg_loss))

        avg_train_loss = train_loss / (train_num)
        avg_train_acc = train_acc / (train_num)

        print(
            "Train-Epoch: {:03d}, Training: Loss: {:.4f}, Train_Accuracy: {:.4f}%".format(
                epoch, avg_train_loss, avg_train_acc * 100))

        w_log.write("Train-Epoch: {:03d}, Training: Loss: {:.4f}, Train_Accuracy: {:.4f}%\n".format(
            epoch, avg_train_loss, avg_train_acc * 100))

        if epoch % 1 == 0:
            cmValue, avg_test_acc, avg_test_loss, test_acc, test_num = computeTestSetAccuracy(model, test_data,
                                                                                              loss_function, arg)
            print(
                "Test-Epoch: {:03d}, Testing: Loss: {:.4f}, Test_Accuracy: {:.4f}%".format(
                    epoch, avg_test_loss, avg_test_acc * 100))
            w_log.write("Test-Epoch: {:03d}, Testing: Loss: {:.4f}, Test_Accuracy: {:.4f}%\n".format(
                epoch, avg_test_loss, avg_test_acc * 100))
            w_log.write('right_num:' + str(test_acc) + '  ' + 'total_num:' + str(test_num) + '\n')

            if avg_test_acc > best_acc:
                best_acc = avg_test_acc
                plot_confusion_matrix(cmValue, avg_test_acc, epoch, arg, classes=action, normalize=True, title='')
                weight_path = os.path.join(arg.work_dir, 'appearance/weight',
                                           str(epoch + 1) + '_' + str(avg_test_acc) + '.pt')
                torch.save(model.state_dict(), weight_path)
