#coding:utf-8

#0 import modules and generate data sets
import tensorflow as tf
import numpy as np
import matplotlib.pylot as plt
BATCH_SIZE = 30
SEED = 2

# Generate random number based on seed
rdm = np.random.RandomState(seed)
# Return 300 line 2 column matrix, represents 300 groups of coordinate (x0,x1) as input data sets
X = rdm.rand(300, 2)
# pick one line from matrix, if the sum of the squares of two coordinates is less than 2, set Y_ to 1,else set Y_ to 0
# As lable of input data sets (right answer)
Y_ = [[int(x0*x0 + x1*x1 < 2) for (x0,x1) in X]]
# Traverse all the points in Y_, if 1 set 'red' and others set to 'blue'.
Y_c = [['red' if y else 'blue'] for y in Y_]
# shape X and Y_, first param -1 means n, second param means column
X = np.vstack(X),reshape(-1, 2)
Y_ = np.vstack(Y_).reshape(-1, 1)
print X
print Y_
print Y_c

# use plt.scatter to draw every coordinates(x0, x1), use values in Y_c to represents color of the line
plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
plt.show()
# Finished the creation of data sets, and draw the visible points

# define the input, paramater and output, forward propagation 
def get_weight(shape, regularizer):
	w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
	tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w
def get_bias(shape):
	b = tf.Variable(tf.constant(0.01, shape=shape))
	return b

x = tf.placeholder(tf.float32, shape=(None,2))
y = tf.placeholder(tf.float32, shape=(None,1))

w1 = get_weight([2,11],0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(x, w1)+b1) # output pass relu function

w2 = get_weight([11,1],0.01)
b2 = get_bias([1])
y = tf.matmul(y1, w2) + b2 # output layer do not pass activate function

# define loss function
loss_mes = tf.reduce_mean(tf.square(y-y_)) #均方误差的损失函数
loss_total = loss_mes + tf.add_n(tf.get_collection('losses')) #均方误差的损失函数加上每一个正则化w的损失

# define backward propagation func: no regularizer
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mes) #adam优化器，沿着仅，均方误差的损失函数最小的方向，进行优化

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	STEPS = 40000
	for i in range(STEPS):
		start = (i*BATCH_SIZE) % 300
		end = start + BATCH_SIZE
		sess.run(train_step, feed_dict={x:X[start:end], y_:Y_[start:end]})
		if i % 2000 == 0:
			loss_mes_v = sess.run(loss_mes, feed_dict={x:X, y_:Y_})
			print ("After %d steps, loss is: %f" %(i, loss_mes_v))

	# xx和yy在-3到3之间以0.01为步长，生成二维网格坐标点
	xx,yy = np.mgrid[-3:3:0.01, -3:3:.01]
	# 将xx，yy拉直，合并成一个2列的矩阵，得到一个网络坐标点的集合
	grid = np.c_[xx.ravel(), yy.ravel()]
	# 将所有网络坐标点喂给神经网络，probs是输出
	probs = sess.run(y, feed_dict={x:grid})
	# 调整probs的shape成xx的样子
	probs = probs.reshape(xx.shape)
	print "w1:\n", sess.run(w1)
	print "b1:\n", sess.run(b1)
	print "w2:\n", sess.run(w2)
	print "b2:\n", sess.run(b2)

plt.scatter(X[:,0],X[:,1], c=np.squeeze(Y_c))
plt.contour(xx,yy,probs,levels=[.5]) #把坐标xx，yy，和对应的值probs放入contour函数，给所有probs值为0.5的点上色
plt.show() #显示未使用正则化后的豁然点分界线

# define backward propagation func: regularizer
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total) #adam优化器，沿着仅，均方误差的损失函数最小的方向，进行优化

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	STEPS = 40000
	for i in range(STEPS):
		start = (i*BATCH_SIZE) % 300
		end = start + BATCH_SIZE
		sess.run(train_step, feed_dict={x:X[start:end], y_:Y_[start:end]})
		if i % 2000 == 0:
			loss_mes_v = sess.run(loss_mes, feed_dict={x:X, y_:Y_})
			print ("After %d steps, loss is: %f" %(i, loss_mes_v))

	# xx和yy在-3到3之间以0.01为步长，生成二维网格坐标点
	xx,yy = np.mgrid[-3:3:0.01, -3:3:.01]
	# 将xx，yy拉直，合并成一个2列的矩阵，得到一个网络坐标点的集合
	grid = np.c_[xx.ravel(), yy.ravel()]
	# 将所有网络坐标点喂给神经网络，probs是输出
	probs = sess.run(y, feed_dict={x:grid})
	# 调整probs的shape成xx的样子
	probs = probs.reshape(xx.shape)
	print "w1:\n", sess.run(w1)
	print "b1:\n", sess.run(b1)
	print "w2:\n", sess.run(w2)
	print "b2:\n", sess.run(b2)

plt.scatter(X[:,0],X[:,1], c=np.squeeze(Y_c))
plt.contour(xx,yy,probs,levels=[.5]) #把坐标xx，yy，和对应的值probs放入contour函数，给所有probs值为0.5的点上色
plt.show() # 显示使用正则化后的豁然点分界线