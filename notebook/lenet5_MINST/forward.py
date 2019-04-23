#coding:gb2312
import tensorflow as tf

# picture size: 28*28*1
IMAGE_SIZE = 28
NUM_CHANNEL = 1

# Convolutional 1 size 5*5*32
CONV1_SIZE = 5
CONV1_KERNEL_NUM = 32

# Convolutional 2 size 5*5*64
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64

# First Full neural size 512, outputsize 10
FC_SIZE = 512
OUTPUT_NODE = 10

# get random weight first
def get_weight(shape, regularizer):
    # truncated_normal 生成去掉过大偏离点的随机数
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses',
                             tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

# 生成偏执
def get_bias(shape):
    #  生成初始值为0的偏执
    b = tf.Variable(tf.zeros(shape))
    return b

#  求卷积
def conv2d(x, w):
    #  x是1*4的矩阵 [喂入数据量，行分辨率，列分辨率，维度]
    #  w也是1*4 描述核 [行分辨率，列分辨率，维度，核数量]
    #  strides 为卷积核步长描述
    #  padding表示全0填充SAME否则VALID
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

#  最大池化函数
def max_pool_2x2(x):
    #  x是1*4的矩阵 [喂入数据量，行分辨率，列分辨率，维度]
    #  k也是1*4 描述核 [不知道，行分辨率，列分辨率，核数1固定]
    #  strides 为卷积核步长描述
    #  padding表示全0填充SAME否则VALID
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def forward(x, train, regularizer):
    #  得到第一层卷积参数随机值  这里是5*5*1*32个w 32个b
    conv1_w = get_weight(
        [CONV1_SIZE, CONV1_SIZE, NUM_CHANNEL, CONV1_KERNEL_NUM], regularizer)
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    # 先求卷积乘积
    conv1 = conv2d(x, conv1_w)
    # 再加偏执 后激活
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    # 池化
    pool1 = max_pool_2x2(relu1)

    #  得到第2层卷积参数随机值  这里是5*5*1*64个w 64个b
    conv2_w = get_weight(
        [CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    # 步骤解释同1
    conv2 = conv2d(pool1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2)

    # 得到特征选择输出的像素形状
    pool_shape = pool2.get_shape().as_list()
    # 得到特征点个数
    # pool_shape[0] 为一个batch值 1长2宽3深
    node = pool_shape[1] * pool_shape[2] * pool_shape[3]
    # 拉伸为pool2的行数乘 batch行，特征点数为列的数据
    reshaped = tf.reshape(pool2, [pool_shape[0], node])

    # 全连接步骤
    # 得到全连接第一层参数
    fc1_w = get_weight([node, FC_SIZE], regularizer)
    fc1_b = get_bias([FC_SIZE])
    # 计算第一层输出
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    # dropout步骤，如果训练阶段，舍弃50%数据元
    if train:
        fc1 = tf.nn.dropout(fc1, 0.5)

    # 第二层全连接
    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b
    return y
