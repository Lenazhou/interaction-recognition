import torch
import time
import os
from torch.autograd import Variable
from models.utils.appearance_test_ci import computeTestSetAccuracy
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
        print("Epoch: {}/{}".format(epoch+1, arg.epoch))

        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_num = 0
        # inputs[B,3,216,384] boxes[B,15,4] labels[B,105] group[B,105,2] real_in_num[B,] real_box_num[B,] skeleton[B,36,216,384] label_s[B,105]
        for ii, (inputs, boxes, labels, group, real_in_num, real_box_num, skeleton, label_s, dis_flag) in enumerate(
                train_data):
            if len(inputs) != arg.batch_size:
                continue
            action_group = None
            action_label = list()
            action_skeleton_label = list()
            ot_label = list()
            flag_num = list()
            for b in range(arg.batch_size):
                batch_group = group[b, :real_in_num[b], :]
                batch_dis = dis_flag[b, :real_in_num[b]]
                a_g = None
                a_s_l = list()
                for g_id, (g, dis) in enumerate(zip(batch_group, batch_dis)):
                    if dis == 1:
                        a_g = g[None, :] if a_g is None else torch.cat((a_g, g[None, :]), 0)
                        action_label.append(labels[b][g_id])
                        a_s_l.append(label_s[b][g_id])
                    else:
                        ot_label.append(labels[b][g_id])

                flag_num.append(len(a_s_l))
                while len(a_s_l) < 36:
                    a_s_l.append(-1)
                    a_g = torch.Tensor([[-1, -1]]).int() if a_g is None else torch.cat(
                        (a_g, torch.Tensor([[-1, -1]]).int()), 0)
                action_skeleton_label.append(a_s_l)
                action_group = a_g if action_group is None else torch.cat((action_group, a_g), 0)
            action_group = action_group.view((-1, 36, 2))
            action_label = torch.Tensor(action_label)
            action_skeleton_label = torch.Tensor(action_skeleton_label)
            flag_num = torch.Tensor(flag_num)

            inputs = Variable(inputs.float().cuda(device), requires_grad=False)
            boxes = Variable(boxes.float().cuda(device), requires_grad=False)
            action_label = Variable(action_label.int().cuda(device), requires_grad=False)
            action_skeleton_label = Variable(action_skeleton_label.int().cuda(device), requires_grad=False)
            action_group = Variable(action_group.int().cuda(device), requires_grad=False)
            skeleton = Variable(skeleton.float().cuda(), requires_grad=False)
            flag_num = Variable(flag_num.float().cuda(), requires_grad=False)
            real_box_num = Variable(real_box_num.int().cuda(), requires_grad=False)

            optimizer.zero_grad()

            train_num = train_num + len(action_label) + len(ot_label)

            outputs = model(inputs, boxes, action_group, real_box_num, action_skeleton_label, skeleton)  # (train_num,7)
            loss = loss_function(outputs, action_label.long())
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            ret, predictions = torch.max(outputs.data, 1)

            correct_counts = predictions.eq(action_label.data.view_as(predictions))

            acc = torch.sum(correct_counts.type(torch.FloatTensor))
            acc_ot = 0
            for ot in ot_label:
                if ot == 9:
                    acc_ot += 1
            acc += acc_ot

            train_acc += acc.item()
            if ii % 200 == 0 and ii != 0:
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
                                                                                              loss_function,arg)
            print(
                "Test-Epoch: {:03d}, Testing: Loss: {:.4f}, Test_Accuracy: {:.4f}%".format(
                    epoch, avg_test_loss, avg_test_acc * 100))
            w_log.write("Test-Epoch: {:03d}, Testing: Loss: {:.4f}, Test_Accuracy: {:.4f}%\n".format(
                epoch, avg_test_loss, avg_test_acc * 100))
            w_log.write('right_num:' + str(test_acc) + '  ' + 'total_num:' + str(test_num) + '\n')

            if avg_test_acc > best_acc:
                best_acc = avg_test_acc
                plot_confusion_matrix(cmValue, avg_test_acc, epoch, arg,classes=action, normalize=True, title='')
                weight_path = os.path.join(arg.work_dir, 'appearance/weight',
                                           str(epoch + 1) + '_' + str(avg_test_acc) + '.pt')
                torch.save(model.state_dict(), weight_path)