#!/usr/bin/python
#coding:utf-8
import inspect
import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg16():
    # 倒入模型参数
    def __init__(self, vgg16_path=None):
        if vgg16_path is None:
            vgg16_path = os.path.join(os.getcwd(), "vgg16.npy")
            self.data_dict = np.load(vgg16_path, encoding='latin1').item()

    # 复现了网络结构
    def forward(self, images):

        print("build model started")
        start_time = time.time() #记录开始时间
        rgb_scaled = images * 255.0 #逐个像素点乘以255
        # 输入的图片在第三个维度切分为三份，分别付给红绿蓝
        red, green ,blue = tf.split(rgb_scaled,3,3)
        # 逐个样本减去各个通道的像素平均值，按照BGR粘贴为新数据
        bgr = tf.concat([
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2]],3)

        # 根据命名空间逐层复现网络
        # 卷积层输入bgr数据，命名为conv1_1
        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool_2x2(self.conv1_2, "pool1")

        # 卷积 卷积 最大池化
        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool_2x2(self.conv2_2, "pool2")

        # 卷积 卷积 卷积 最大池化
        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool_2x2(self.conv3_3, "pool3")
        
        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool_2x2(self.conv4_3, "pool4")

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool_2x2(self.conv5_3, "pool5")

        # 过一层全连接网络
        self.fc6 = self.fc_layer(self.pool5, "fc6")
        self.relu6 = tf.nn.relu(self.fc6)
        
        # 又过一层全连接网络
        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, "fc8")
        #输出过softmax函数，得到千分类概率分布
        self.prob = tf.nn.softmax(self.fc8, name="prob")
        
        # 计算数据走过整个网络的时间消耗
        end_time = time.time()
        print(("time consuming: %f" % (end_time-start_time)))

        # 清空本次读取到的模型参数
        self.date_dict = None

    # 卷积核参数读取
    def conv_layer(self, x, name):
        with tf.variable_scope(name):
            w = self.get_conv_filter(name)
            conv = tf.nn.conv2d(x, w, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name)
            result = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
            return result

    #卷积偏置参数读取
    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    #卷积偏置参数读取
    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    # 最大池化
    def max_pool_2x2(self, x, name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)

    # 全连接网络计算
    def fc_layer(self, x, name):
        with tf.variable_scope(name):       # 建立全连接层的命名空间
            shape = x.get_shape().as_list() # 获取该层的维度信息
            dim = 1
            for i in shape[1:]:
                dim *= i
            x = tf.reshape(x, [-1, dim])    # 将得到的多维信息拉直
            w = self.get_fc_weight(name)
            b = self.get_bias(name)
            
            # 对该层加权求和，再加上偏执b
            result = tf.nn.bias_add(tf.matmul(x, w), b)
            return result
    
    # 定义获取全连接权重的函数
    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")
















