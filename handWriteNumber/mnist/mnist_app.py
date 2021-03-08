# coding:utf-8

import tensorflow as tf
import numpy as np
import matplotlib as plt
from PIL import Image
import mnist_backward
import mnist_forward

def restore_model(testPicArr):
    # 重现计算图
    with tf.Graph().as_default() as tg:
        # 仅需要给输入x占位
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)
        # y的最大值对应的列表索引号，就是预测结果
        preValue = tf.argmax(y, 1)

        # 实例化带有滑动平均值得saver
        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                # 如果ckpt存在，恢复ckpt的参数等信息到当前会话sess
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 图片魏如网络，执行预测操作
                preValue = sess.run(preValue, feed_dict={x:testPicArr})
                return preValue
            else:
                # 如果没有找到ckpt，给出提示
                print "No checkpoint file found"
                return -1

def pre_pic(picName):
    # open picture
    img = Image.open(picName)
    # 用消除锯齿的方法resize
    reIm = img.resize((28,28), Image.ANTIALIAS)
    # plt.figure(picName)
    # plt.imshow(reIm)
    # plt.show()

    # 变成灰度图，转换成矩阵
    im_arr = np.array(reIm.convert('L'))
    threshold = 50
    # 反色,降低噪声
    for i in range(28):
       for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if  im_arr[i][j] < threshold:
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255

    nm_arr = im_arr.reshape([1,784])
    # 模型要求为浮点数
    nm_arr = nm_arr.astype(np.float32)
    # 变成0～1之间的浮点数
    img_ready = np.multiply(nm_arr, 1.0/255.0)

    return img_ready

def application():
    testNum = input("input the number of test pictures:")
    for i in range(testNum):
        # read string from cmd
        testPic = raw_input("the path of the test picture:")
        testPicArr = pre_pic(testPic)
        preValue = restore_model(testPicArr)
        print "The prediction number is:", preValue

def main():
    application()

if __name__ == '__main__':
    main()
