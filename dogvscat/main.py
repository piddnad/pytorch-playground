# coding:utf-8
from config import opt
import os
import torch
import models
from data.dataset import DogCat
from torch.utils.data import DataLoader
from torchnet import meter
from utils.visualize import Visualizer
from tqdm import tqdm
import numpy as np
from torchvision import models
from torch import nn
import time
from torchvision import transforms as T

@torch.no_grad()  # pytorch>=0.5
def test(**kwargs):
    opt._parse(kwargs)

    # configure model
    # model = getattr(models, opt.model)().eval()
    # if opt.load_model_path:
    #     model.load(opt.load_model_path)
    model1 = models.resnet101().eval()
    class_num = 2  # 要分类数目是2
    channel_in = model1.fc.in_features  # 获取fc层的输入通道数
    model1.fc = nn.Linear(channel_in, class_num)

    model1.load_state_dict(torch.load('checkpoints/resnet101_0326_1429_98.82666666666667.pth'))
    model1.to(opt.device)

    model2 = models.inception_v3().eval()
    class_num = 2  # 要分类数目是2
    channel_in = model2.fc.in_features  # 获取fc层的输入通道数
    model2.fc = nn.Linear(channel_in, class_num)

    model2.load_state_dict(torch.load('checkpoints/InceptionV3_0326_1548_98.42666666666666.pth'))
    model2.to(opt.device)

    # data
    test_data1 = DogCat(opt.test_data_root, test=True, transforms=1)
    test_dataloader1 = DataLoader(test_data1, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)


    test_data2 = DogCat(opt.test_data_root, test=True)
    test_dataloader2 = DataLoader(test_data2, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)


    results1 = []
    for ii, (data, path) in tqdm(enumerate(test_dataloader1)):
        input = data.to(opt.device)
        score1 = model1(input)
        probability = torch.nn.functional.softmax(score1, dim=1)[:, 0].detach().tolist()
        # label = score.max(dim = 1)[1].detach().tolist()
        probability = np.clip(probability, 0.005, 0.995) # added in 2019/3/22
        batch_results = [(path_.item(), probability_) for path_, probability_ in zip(path, probability)]

        results1 += batch_results

    results2 = []
    for ii, (data, path) in tqdm(enumerate(test_dataloader2)):
        input = data.to(opt.device)
        score2 = model2(input)
        probability = torch.nn.functional.softmax(score2, dim=1)[:, 0].detach().tolist()
        # label = score.max(dim = 1)[1].detach().tolist()
        probability = np.clip(probability, 0.005, 0.995) # added in 2019/3/22

        batch_results = [(path_.item(), probability_) for path_, probability_ in zip(path, probability)]

        results2 += batch_results

    vec_avg = lambda a, b: tuple([(x + y) / 2 for x, y in zip(a, b)])
    avgresults = [vec_avg(a, b) for a, b in zip(results1, results2)]


    write_csv(avgresults, opt.result_file)

    return avgresults


def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


def train(**kwargs):
    torch.cuda.set_device(1)
    opt._parse(kwargs)
    vis = Visualizer(opt.env, port=opt.vis_port)

    # step1: configure model
    # model = getattr(models, opt.model)()
    # if opt.load_model_path:
    #     model.load(opt.load_model_path)
    # model.to(opt.device)
    # resnet101 = models.resnet101(pretrained=True).to(opt.device)
    model = models.inception_v3(pretrained=True).to(opt.device)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    class_num = 2  # 要分类数目是2
    channel_in = model.fc.in_features  # 获取fc层的输入通道数
    model.fc = nn.Linear(channel_in, class_num).to(opt.device)


    # step2: data
    train_data = DogCat(opt.train_data_root, train=True)
    val_data = DogCat(opt.train_data_root, train=False)
    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size,
                                shuffle=False, num_workers=opt.num_workers)

    # step3: criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = model.get_optimizer(lr, opt.weight_decay)
    lr = opt.lr # 后边要用！
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

    # step4: meters
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e10

    # train
    for epoch in range(opt.max_epoch):

        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label) in tqdm(enumerate(train_dataloader)):

            # train model
            input = data.to(opt.device)
            target = label.to(opt.device)

            optimizer.zero_grad()
            score, aux = model(input) # aux: inception_v3 only!!
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            # meters update and visualize
            loss_meter.add(loss.item())
            # detach 一下更安全保险
            confusion_matrix.add(score.detach(), target.detach())

            if (ii + 1) % opt.print_freq == 0:
                vis.plot('loss', loss_meter.value()[0])

                # 进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb;
                    ipdb.set_trace()

        # validate and visualize
        val_cm, val_accuracy = val(model, val_dataloader)

        # model.save()
        prefix = 'checkpoints/' + 'InceptionV3' + '_'
        name = time.strftime(prefix + '%m%d_%H%M_' + val_accuracy.__str__() + '.pth')
        torch.save(model.state_dict(), name)

        vis.plot('val_accuracy', val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
            lr=lr))

        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]


@torch.no_grad()
def val(model, dataloader):
    """
    计算模型在验证集上的准确率等信息
    """
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, (val_input, label) in tqdm(enumerate(dataloader)):
        val_input = val_input.to(opt.device)
        score = model(val_input)
        confusion_matrix.add(score.detach().squeeze(), label.type(torch.LongTensor))

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy


def help():
    """
    打印帮助的信息： python file.py help
    """

    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    import fire

    fire.Fire()