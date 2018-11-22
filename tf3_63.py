#coding:utf-8

import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
SEED = 23455

rng = np.random.RandomState(SEED)
X = rng.rand(32, 2)
Y_ = [[int(x0 + x1 < 1)] for (x0, x1) in X]

print "X:\n", X
print "Y_:\n", Y_

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)

	loss_mes = tf.reduce_mean(tf.square(y-y_))
	train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mes)

	STEPS = 3000
	for i in range(STEPS):
		start = (i * BATCH_SIZE) % 32
		end = start + BATCH_SIZE
		
		sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})

		if i % 500 == 0:
			total_loss = sess.run(loss_mes, feed_dict={x: X, y_:Y_})
			print "After %d steps training, the total loss is %g" % (i, total_loss)

	print "w1:\n", sess.run(w1)
	print "w2:\n", sess.run(w2)
	print "\n"
