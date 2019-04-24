#coding:utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import vgg16
import utils
from Nclasses import labels

img_path = input('Input the path and image name:')
img_ready = utils.load_image(img_path)
# 打印出图片维度
# print(tf.Session(graph=tf.Graph()).run(img_ready))

fig=plt.figure(u"Top-5 预测结果")

with tf.Session() as sess:
    # 图片占位
    images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    # 如果参数已经有了，读出参数
    vgg = vgg16.Vgg16()
    # 前向传播函数
    vgg.forward(images)
    # 获得输出概率结果
    probability = sess.run(vgg.prob, feed_dict={images:img_ready})
    # 得到前五值
    top5 = np.argsort(probability[0])[-1:-6:-1]
    print("top5:",top5)
    # 存概率值
    values = []
    # 存标签字典对应值
    bar_label = []
    for n, i in enumerate(top5):
        # 打印键以及值
        print ("n:",n)
        print ("i:",i)
        # 保存键和标签
        values.append(probability[0][i])
        bar_label.append(labels[i])
        print (i, ":", labels[i], "----", utils.percent(probability[0][i]))

    # 汇出图
    ax = fig.add_subplot(111)
    ax.bar(range(len(values)), values, tick_label=bar_label, width=0.5, fc='g')
    ax.set_ylabel(u'probabilityit')
    ax.set_title(u'Top-5')
    # 添加柱状图注释
    for a,b in zip(range(len(values)), values):
        ax.text(a, b+0.0005, utils.percent(b), ha='center', va = 'bottom', fontsize=7)
    plt.show()
