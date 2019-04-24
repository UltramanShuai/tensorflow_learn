#!/usr/bin/python
#coding:utf-8
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pylab import mpl

mpl.rcParams['font.sans-serif']=['SimHei'] # 正常显示中文标签
mpl.rcParams['axes.unicode_minus']=False # 正常显示正负号

def load_image(path):
    # 新建图底
    fig = plt.figure("Centre and Resize")
    # 传入图片
    img = io.imread(path)
    # 像素归一化
    img = img / 255.0

    # 建立一行三列子图
    ax0 = fig.add_subplot(131)
    # 添加子图标签
    ax0.set_xlabel(u'Original Picture')
    # 画出
    ax0.imshow(img)

    # 找出最短边
    short_edge = min(img.shape[:2])
    # 长宽减去最短边并且除2 求出中心点
    y = int((img.shape[0] - short_edge) / 2)
    x = int((img.shape[1] - short_edge) / 2)
    # 取出切分后的中心图
    crop_img = img[y:y+short_edge, x:x+short_edge]

    # 图像放入第二个图位
    ax1 = fig.add_subplot(132)
    ax1.set_xlabel(u"Centre Picture")
    ax1.imshow(crop_img)

    # 图片重新定型为224*224分辨率
    re_img = transform.resize(crop_img, (224, 224))

    # 放入第三个图位
    ax2 = fig.add_subplot(133)
    ax2.set_xlabel(u"Resize Picture")
    ax2.imshow(re_img)

    # 图片调整成1*224*224*3
    img_ready = re_img.reshape((1, 224, 224, 3))

    return img_ready

def percent(value):
    return '%.2f%%' % (value * 100)
