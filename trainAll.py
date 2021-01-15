from dataset import MyDataset
import torch
import torch.nn.functional as F
from net import Lankmark_net
from mnn import MNN, MNN1D
from torchvision import transforms

import torch.optim as optim
from torch.autograd import Variable
from torch import tensor
#from CNN import resnet1
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import argparse
import os
import logging
from models import *

def getModel(index):
    if index == 1:
        return MNN()
    if index == 2:
        return MNN1D()


use_cuda = torch.cuda.is_available()
print(use_cuda)
#use_cuda = False
for fold_index in range(10):
    best_Test_acc = 0  # best PrivateTest accuracy
    best_Test_acc_epoch = 0
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    total_epoch = 100


    transform = transforms.Compose([transforms.ToTensor()])

    trainset = MyDataset(split='Training', fold=fold_index, transform=transform)
    testset = MyDataset(split='Testing', fold=fold_index, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=75, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=0)

    model_index = 2

    if use_cuda:
        #model = Lankmark_net().cuda()
        model = getModel(model_index).cuda()
    else:
        model = getModel(model_index).cuda()

    print(model)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.MSELoss()
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    log_file_name = './log/ck+_log1.log'
    logging.basicConfig(level=logging.INFO,
                        filename=log_file_name,
                        filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s  %(message)s")


    def train(epoch):
        print('\nEpoch: %d' % epoch)
        global Train_acc
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs = inputs[1].type(torch.FloatTensor)
            #inputs = inputs[0]
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            # inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            print('Train Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                  (train_loss, float(correct) / float(total) * 100, correct, total))
        Train_acc = 100.*correct/total
        train_acc_list.append(Train_acc)
        train_loss_list.append(train_loss)


    def test(epoch):
        global Test_acc
        global best_Test_acc
        global best_epoch
        model.eval()
        PrivateTest_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs = inputs[1].type(torch.FloatTensor)
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                PrivateTest_loss += loss.data.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()
                print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                      (PrivateTest_loss, float(correct)/float(total)*100, correct, total))
        Test_acc = 100.*correct/total
        print('                                            Test_acc: ', float(Test_acc))
        test_loss_list.append(loss)
        test_acc_list.append(Test_acc)
        if Test_acc > best_Test_acc:
            best_Test_acc = Test_acc
            best_epoch = epoch
        print("                                                                           epoch:", epoch, " best_Test_acc_epoch: ", int(best_epoch),
              " best_Test_acc: ", float(best_Test_acc))

        if epoch == total_epoch-1:
            logging.info('best_test_acc: {0}, best_epoch:{1}, model:{2}, fold:{3}'.format(best_Test_acc,best_epoch,model_index,opt.fold))
        '''
        if epoch == total_epoch - 1:
            best_Test_acc_fold = best_Test_acc
        '''


    def acc_loss_plot(list1, list2, list3, list4, total_epoch, every):
        x1 = range(0, total_epoch, every)
        x2 = range(0, total_epoch, every)
        x3 = range(0, total_epoch, every)
        x4 = range(0, total_epoch, every)
        y1 = list1[::every]
        y2 = list2[::every]
        y3 = list3[::every]
        y4 = list4[::every]
        plt.subplot(2, 2, 1)
        plt.title("train_acc")
        plt.plot(x1, y1)
        plt.subplot(2, 2, 2)
        plt.title("train_loss")
        plt.plot(x2, y2)
        plt.subplot(2, 2, 3)
        plt.title('test_acc')
        plt.plot(x3, y3)
        plt.subplot(2, 2, 4)
        plt.title("test_loss")
        plt.plot(x4, y4)
        timing = time.strftime("%Y%m%d%H%M%S", time.localtime())
        plt.savefig("./plot/acc_loss_plot %s.jpg" % timing)
        # plt.show()

    def run():
        for epoch in range(start_epoch, total_epoch):
            train(epoch)
            test(epoch)
        acc_loss_plot(train_acc_list, train_loss_list,
                      test_acc_list, test_loss_list, total_epoch, 2)




if __name__ == '__main__':
    run()
