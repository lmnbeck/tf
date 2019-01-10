#coding:utf-8
# 描述网络结构
import tensorflow as tf

IMAGE_SIZE = 28
NUM_CHANNELS = 1
CONV1_SIZE = 5
CONV1_KERNEL_NUM =32
CONV2_SIZE = 5
CONV2_KERNEL_NUM =64
FC_SIZE = 512
OUTPUT_NODE = 10

def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def forward(x, regularizer):
    //初始化第一层卷积核
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)
    //初始化第一层偏置
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    //执行卷积计算
    conv1 = conv2d(x, conv1_w)
    //对卷积后的输出conv1添加偏置conv1_b, 通过relu激活函数
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    //激活后的输出进行最大池化
    pool1 = max_pool_2X2(relu1)

    //初始化第二层卷积核
    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
    //初始化第二层偏置
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    //执行卷积计算，输入是上层的pool1
    conv2 = conv2d(pool1, conv2_w)
    //对卷积后的输出conv1添加偏置conv1_b, 通过relu激活函数
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    //激活后的输出进行最大池化. pool2是第二层的输出，需要把它从三维张量变为二维张量
    pool2 = max_pool_2X2(relu2)

    //得到pool2输出矩阵的维度，存入list中
    pool_shape = pool2.get_shape().as_list()
    //提取特征的长度，宽度，深度，相乘得到所有特征点的个数
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    //将pool2表示成batch行，所有特征点作为列的二维形状。把它再喂到全连接网络里
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    fc1_w = get_weight([nodes, FC_SIZE], regularizer)
    fc1_b = get_bias([FC_SIZE])
    //上层的输出乘以本层的权重，加上偏置，过激活函数relu
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    //如果是训练阶段，对该层的输出使用50%的dropout
    if train: fc1 = tf.nn.dropout(fc1, 0.5)
    
    //初始化第二层
    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    
    y = tf.matmul(fc1, fc2_w) + fc2_b
    return y
