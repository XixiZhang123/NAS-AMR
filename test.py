import os
import sys
import numpy as np
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from thop import profile
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from model import NetworkImageNet as Network
import itertools
from sklearn.metrics import *

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='/data/zhangxx/package/TFDDatabase-20-30(xx)',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=2, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=4, help='total number of layers')  # 14初始
parser.add_argument('--model_path', type=str,
                    default='PC-DARTS 20(6nodes,init_16,128)(k=4) qeval-EXP-20211021-202354/model_best.pth.tar',
                    help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='PCDARTS', help='which architecture to use')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CLASSES = 8
classes = ['Barcker', 'B-FM', 'BPSK', 'CPFSK', 'DSB-AM', 'GFSK', 'LFM', 'Rect']


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda()
    model.load_state_dict(torch.load(args.model_path)['state_dict'], False)
    model.drop_path_prob = args.drop_path_prob
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    input = torch.cuda.FloatTensor(1, 3, 128, 128)
    # macs, params = profile(model, inputs=(input,))
    # macs, params = clever_format([macs, params], "%.3f")
    # print("flops    params")
    # print(macs, params)
    # summary(model, input_size=(1, 2, 128))
    flops, params = profile(model, inputs=(input,))
    print("flops")
    print(flops)
    print(params)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    validdir = os.path.join(args.data, 'test20')
    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
    valid_data = dset.ImageFolder(
        validdir,
        transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            normalize,
        ]))

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    model.drop_path_prob = args.drop_path_prob
    test_acc, test_obj, target, loggg, probabily_list, target_list = infer(valid_queue, model, criterion)
    logging.info('test_acc %f', test_acc)

    probabily_data_list = []
    predict_label_list = []
    for batch_i in probabily_list:
        for data_i in batch_i:
            # print(data_i)
            probabily_data_list.append(data_i)

            # max_value, index = max(data_i),data_i.index(max(data_i))
            max_value = max(data_i)
            max_index = np.where(data_i == max_value)
            max_index = max_index[0].tolist()[0]
            # print(data_i,max_value,max_index)
            predict_label_list.append(max_index)

    target_data_list = []
    for batch_i in target_list:
        for data_i in batch_i:
            # print(data_i)
            target_data_list.append(data_i)

    '''
    Count Accuracy
      # real label vs predict label
  # real_label = target_data_list
  # predict_label = predict_label_list

  '''
    print("target_data_list = ", target_data_list)
    print("predict_label_list = ", predict_label_list)

    #accuracy = accuracy_score(target_data_list, predict_label_list)
    #print("accuracy = ", accuracy)

    precision = precision_score(target_data_list, predict_label_list, average='macro')
    print("precision = ", precision)

    recall = recall_score(target_data_list, predict_label_list, average='macro')
    print("recall = ", recall)

    f1 = f1_score(target_data_list, predict_label_list, average='macro')
    print("f1 = ", f1)

    # cm = confusion_matrix(target, loggg)
    #
    # print(cm)
    # plot_confusion_matrix(cm, classes, title='Confusion matrix of 0dB', cmap=plt.cm.Blues)


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    probabily_list = []
    target_list = []
    max_value_list = []  #
    prob_class_list = []
    for step, (input, target_1) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target_1, volatile=True).cuda(async=True)

        logits, _ = model(input)

        # print(logits.shape)
        if step == 0:
            FTA = target
            probabily = torch.nn.functional.softmax(logits, dim=1)
            max_value, index = torch.max(probabily, 1)
            FTL = index
            loss = criterion(logits, target)

            #
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            # print(" step = 0")

            if step % args.report_freq == 0:
                logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        else:
            # print(" else of step = 0 ")
            FTA = torch.cat((FTA, target), dim=0)

            probabily = torch.nn.functional.softmax(logits, dim=1)
            max_value, index = torch.max(probabily, 1)
            FTL = torch.cat((FTL, index), dim=0)
        probabily = probabily.cpu().detach().numpy()
        probabily_list.append(probabily)

        target_list.append(target_1.numpy())

        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg, FTA, FTL, probabily_list, target_list  # ,max_value_list,prob_class_list


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  ###????

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)  # rotation=45 x?????45?
    plt.yticks(tick_marks, classes)
    plt.tight_layout(pad=2)

    ###????
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    ###????

    '''
  iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
  # # ij??????????
  iters = np.reshape([[[i, j] for j in range(4)] for i in range(4)], (cm.size, 2))
  for i, j in iters :
      plt.text(j, i, format(cm[i, j]))  # ???????
  '''
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == '__main__':
    main()

