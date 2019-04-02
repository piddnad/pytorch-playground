#coding:utf-8
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T

class DogCat(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):
        """
        获取所有图片地址，并划分数据为训练、验证、测试集
        :param root:
        :param transforms:
        :param train:
        :param test:
        """
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # test1: ~/datasets/dogvscat/test1/8973.jpg
        # train: ~/datasets/dogvscat/train/cat.10004.jpg
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

        imgs_num = len(imgs)

        # 划分训练、验证集 = 7:3
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]

        if transforms is None:
            # 数据变换
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            # 验证集和测试集
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(299),
                    T.CenterCrop(299),
                    T.ToTensor(),
                    normalize
                ])
            # 训练集
            else:
                self.transforms = T.Compose([
                    T.Resize(299),
                    T.RandomResizedCrop(299),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])
        else:
            self.transforms = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        """
        返回一张图片的数据
        对于测试集，没有label，返回图片id
        :param index:
        :return:
        """
        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)