import torch
from torchnet import meter
from torch.autograd import Variable
import tqdm


def computeTestSetAccuracy(model, test_data, loss_function, arg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_acc = 0.0
    test_loss = 0.0
    test_num = 0

    action_tot_num = 0.0
    action_acc_num = 0.0

    with torch.no_grad():
        model.eval()
        confusion_matrix = meter.ConfusionMeter(arg.num_class)
        for j, (inputs, boxes, labels, group, real_in_num, real_box_num, skeleton, label_s, dis_flag) in enumerate(
                test_data):
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
                    a_g = torch.cat((a_g, torch.Tensor([[-1, -1]]).int()), 0)
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

            test_num = test_num + len(action_label) + len(ot_label)
            outputs = model(inputs, boxes, action_group, real_box_num, action_skeleton_label, skeleton)
            loss = loss_function(outputs, action_label.long())

            test_loss += loss.item()
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(action_label.data.view_as(predictions))

            acc = torch.sum(correct_counts.type(torch.FloatTensor))

            for p_l, g_l in zip(predictions, action_label):
                if g_l != 9:
                    action_tot_num += 1
                    if p_l == g_l:
                        action_acc_num += 1

            acc_ot = 0
            for ot in ot_label:
                if ot == 9:
                    acc_ot += 1
            acc += acc_ot

            test_acc += acc.item()
            confusion_matrix.add(outputs.squeeze(), action_label.long())

            if j % 200 == 0 and j != 0:
                avg_acc = test_acc / test_num
                avg_loss = test_loss / test_num
                print("Test dataloader ii : {}, avg_acc : {:.4f}%, avg_loss: {:.4f}".format(j, avg_acc * 100, avg_loss))
    avg_test_loss = test_loss / (test_num)
    avg_test_acc = test_acc / (test_num)
    avg_action_ot = action_acc_num / action_tot_num
    print('Action no OT Acc: ' + str(avg_action_ot * 100) + '%')
    cmValue = confusion_matrix.value()
    return cmValue, avg_test_acc, avg_test_loss, test_acc, test_num