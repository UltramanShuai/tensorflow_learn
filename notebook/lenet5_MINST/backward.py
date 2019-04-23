#coding:gb2312
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import forward
import os
import numpy as np

# 喂入训练数量
BATCH_SIZE = 100
# 学习率初始值
LEARNING_RATE_BASE = 0.005
# 学习率下降后比例
LEARNING_RATE_DECAY = 0.99
# 泛化值
REGULARIZER = 0.0001
# 训练次数
STEPS = 50000
# 滑动平均下降后比例
MOVING_AVERAGE_DECAY = 0.99
# 模型储存路径
MODEL_SAVE_PATH = "./model/"
# 模型储存名字
MODEL_NAME = "mnist_model"


def backward(mnist):
    # x按照图片给出
    x = tf.placeholder(tf.float32, [
                       BATCH_SIZE, forward.IMAGE_SIZE, forward.IMAGE_SIZE, forward.NUM_CHANNEL])
    y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])

    y = forward.forward(x, True, REGULARIZER)

    # 全局计步
    global_step = tf.Variable(0, trainable=False)

    # 交叉熵
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    # 损失函数
    loss = cem + tf.add_n(tf.get_collection('losses'))

    # 学习率消减
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY, staircase=True
    )

    # 梯度下降
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss, global_step=global_step)

    # 滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 对所有训练数据应用
    ema_op = ema.apply(tf.trainable_variables())

    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    # 设置保存
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 初始化参数
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 检查是否存在模型，存在就提取
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshape_xs = np.reshape(
                xs, (BATCH_SIZE, forward.IMAGE_SIZE, forward.IMAGE_SIZE, forward.NUM_CHANNEL))

            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: reshape_xs, y_: ys})

            if i % 100 == 0:
                print("After %d training step(s)m loss on training batch is %g." % (
                    step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                           global_step=global_step)


def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    backward(mnist)


if __name__ == '__main__':
    main()
