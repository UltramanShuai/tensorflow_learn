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
    # truncated_normal ����ȥ������ƫ���������
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses',
                             tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

# ����ƫִ
def get_bias(shape):
    #  ���ɳ�ʼֵΪ0��ƫִ
    b = tf.Variable(tf.zeros(shape))
    return b

#  ����
def conv2d(x, w):
    #  x��1*4�ľ��� [ι�����������зֱ��ʣ��зֱ��ʣ�ά��]
    #  wҲ��1*4 ������ [�зֱ��ʣ��зֱ��ʣ�ά�ȣ�������]
    #  strides Ϊ����˲�������
    #  padding��ʾȫ0���SAME����VALID
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

#  ���ػ�����
def max_pool_2x2(x):
    #  x��1*4�ľ��� [ι�����������зֱ��ʣ��зֱ��ʣ�ά��]
    #  kҲ��1*4 ������ [��֪�����зֱ��ʣ��зֱ��ʣ�����1�̶�]
    #  strides Ϊ����˲�������
    #  padding��ʾȫ0���SAME����VALID
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def forward(x, train, regularizer):
    #  �õ���һ�����������ֵ  ������5*5*1*32��w 32��b
    conv1_w = get_weight(
        [CONV1_SIZE, CONV1_SIZE, NUM_CHANNEL, CONV1_KERNEL_NUM], regularizer)
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    # �������˻�
    conv1 = conv2d(x, conv1_w)
    # �ټ�ƫִ �󼤻�
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    # �ػ�
    pool1 = max_pool_2x2(relu1)

    #  �õ���2�����������ֵ  ������5*5*1*64��w 64��b
    conv2_w = get_weight(
        [CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    # �������ͬ1
    conv2 = conv2d(pool1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2)

    # �õ�����ѡ�������������״
    pool_shape = pool2.get_shape().as_list()
    # �õ����������
    # pool_shape[0] Ϊһ��batchֵ 1��2��3��
    node = pool_shape[1] * pool_shape[2] * pool_shape[3]
    # ����Ϊpool2�������� batch�У���������Ϊ�е�����
    reshaped = tf.reshape(pool2, [pool_shape[0], node])

    # ȫ���Ӳ���
    # �õ�ȫ���ӵ�һ�����
    fc1_w = get_weight([node, FC_SIZE], regularizer)
    fc1_b = get_bias([FC_SIZE])
    # �����һ�����
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    # dropout���裬���ѵ���׶Σ�����50%����Ԫ
    if train:
        fc1 = tf.nn.dropout(fc1, 0.5)

    # �ڶ���ȫ����
    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b
    return y
