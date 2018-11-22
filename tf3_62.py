#coding:utf-8

import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
SEED = 23455

rdm = np.random.RandomState(SEED)
X =rdm.rand(32,2)
Y_ = [[int(x0+x1<1)] for (x0,x1) in X]
print "X:\n",X
print "Y_:\n",Y_


