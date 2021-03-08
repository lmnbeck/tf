#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import vgg16
import utils
from Nclasses import labels

img_path = raw_input('Input the path and image name:')
# 对图片进行预处理
img_ready = utils.load_image(img_path)
# print打印出

# 结果打印到柱状图上
fig = plt.figure(u"Top-5 预测结果")

with tf.Session() as sess:
    images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    vgg = vgg16.Vgg16()
    vgg.forward(images)
    #生成概率分布
    probability = sess.run(vgg.prob, feed_dict={images:img_ready})
    top5 = np.argsort(probability[0])[-1:-6:-1]
    print "top5:",top5

    # 存probability元素值
    values = []
    # 存标签字典中对应的值
    bar_label = []
    for n, i in enumerate(top5):
        print "n:",n
        print "i:",i
        values.append(probability[0][i])
        bar_label.append(labels[i])
        print i, ":", labels[i], "----", utils.percent(probability[0][1])

    # 画出柱状图
    ax = fig.add_subplot(111)
    ax.bar(range(len(values)), values, tick_label=bar_label, width=0.5, fc='g')
    ax.set_ylabel(u'probabilitiyit')
    ax.set_title(u'Top-5')
    for a,b in zip(range(len(values)), values):
        ax.text(a, b+0.0005, utils.percent(b), ha='center', va='bottom', fontsize=7)
    plt.show()
        


