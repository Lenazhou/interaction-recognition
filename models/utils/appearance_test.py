import torch
from torchnet import meter
from torch.autograd import Variable
import tqdm

def computeTestSetAccuracy(model,test_data, loss_function,arg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_acc = 0.0
    test_loss = 0.0
    test_num=0
    with torch.no_grad():
        model.eval()
        confusion_matrix = meter.ConfusionMeter(arg.num_class)
        for j, (inputs, boxes, labels, group, real_in_num, real_box_num,skeleton,label_s) in enumerate(test_data):
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
            skeleton = Variable(skeleton.float().cuda(),requires_grad=False)
            test_num += len(labels_real)

            outputs = model(inputs, boxes, group, real_in_num, label_s,skeleton)
            loss = loss_function(outputs, labels_real.long())


            test_loss += loss.item()

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels_real.data.view_as(predictions))

            acc = torch.sum(correct_counts.type(torch.FloatTensor))

            test_acc += acc.item()
            confusion_matrix.add(outputs.squeeze(), labels_real.long())
            if j % 50 == 0 and j!=0:
                avg_acc = test_acc/test_num
                avg_loss = test_loss/test_num
                print("Test dataloader ii : {}, avg_acc : {:.4f}%, avg_loss: {:.4f}".format(j, avg_acc * 100,avg_loss))

    avg_test_loss = test_loss / (test_num)
    avg_test_acc = test_acc / (test_num)

    cmValue=confusion_matrix.value()
    return cmValue,avg_test_acc,avg_test_loss,test_acc,test_num
