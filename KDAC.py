# -*- codeing = UTF-8 -*-
# @Time : 2021/8/10 13:17
# @Author : wzh
# @File : KDAC.py

import tensorflow as tf
import numpy as np

class Kdac(object):
    def __init__(self):
        super().__init__()
        self.alpha = tf.Variable(0.1, trainable=True)
        self.beta = tf.Variable(0.1, trainable=True)

    def smin(self,a,b):
        h = tf.clip_by_value(0.5 + 0.5 * (a-b) / self.k, 0, 1)
        return a * (1-h) + h * b - self.k * h * (1-h)

    def smax(self,a,b):
        h = tf.clip_by_value(0.5 - 0.5 * (a-b) / self.k, 0, 1)
        return a * (1-h) + h * b + self.k * h * (1-h)

    def negative_region(self,x):
        f1 = self.alpha * x
        f2 = self.smin(self.beta * x, tf.tanh(x))
        condition = tf.greater(x, 0)
        ne = tf.where(condition, f1, f2)
        return ne

    def positive_region(self,x):
        po = self.alpha * x
        return po

    def kdac(self, x):
        self.k = 0.01
        ac = self.smax(self.positive_region(x), self.negative_region(x))

        return ac

