
import tensorflow as tf
from os import walk
import os
import numpy as np
from util import get_wav, to_spec
from model import infer
from config import NetConfig_MIR_1K
from random import *
import math

#load data
trainMixed = []
trainSrc1 = []
trainSrc2 = []
trainNum = 0
batchSize = 4

print('generate train spectrograms')

for (root, dirs, files) in walk(NetConfig_MIR_1K.DATA_PATH):
    for f in files:
        if f.startswith("abjones") or f.startswith("amy"):
            filename = '{}/{}'.format(root, f)
            mixed_wav, src1_wav, src2_wav = get_wav(filename)
            mixed_spec = to_spec(mixed_wav)
            src1_spec = to_spec(src1_wav)
            src2_spec = to_spec(src2_wav)
            mixed_spec_mag = np.abs(mixed_spec)
            src1_spec_mag = np.abs(src1_spec)
            src2_spec_mag = np.abs(src2_spec)

            maxVal= np.max(mixed_spec_mag)
            mixed_spec_mag = mixed_spec_mag / maxVal
            src1_spec_mag = src1_spec_mag / maxVal
            src2_spec_mag = src2_spec_mag / maxVal

            if (mixed_spec_mag.shape[-1]) < 64:
                pad_len = math.ceil((64-mixed_spec_mag.shape[-1])/2)
                print(mixed_spec_mag.shape[-1])
                print(pad_len)
                trainMixed.append(np.pad(mixed_spec_mag,((0, 0), (pad_len, pad_len)),'constant'))
                trainSrc1.append(np.pad(src1_spec_mag,((0, 0), (pad_len, pad_len)),'constant'))
                trainSrc2.append(np.pad(src2_spec_mag,((0, 0), (pad_len, pad_len)),'constant'))
            else:
                trainMixed.append(mixed_spec_mag)
                trainSrc1.append(src1_spec_mag)
                trainSrc2.append(src2_spec_mag)
            trainNum = trainNum+1

print('Number of training examples : {}'.format(trainNum))

# Model
print('Initialize network')
with tf.device('/device:GPU:0'):
    y_output=[]
    x_mixed = tf.placeholder(tf.float32, shape=(batchSize, 512, 64, 1), name='x_mixed')
    y_mixed = tf.placeholder(tf.float32, shape=(batchSize, 512, 64, 2), name='y_mixed')
    y_pred = infer(x_mixed,2)
    net = tf.make_template('net',y_pred)
    y_output.append(tf.multiply(x_mixed,y_pred[0]))
    loss_0 = tf.reduce_mean(tf.abs(y_mixed - y_output[0]) , name='loss0')
    y_output.append(tf.multiply(x_mixed,y_pred[1]))
    loss_1 = tf.reduce_mean(tf.abs(y_mixed - y_output[1]) , name='loss1')
    y_output.append(tf.multiply(x_mixed,y_pred[2]))
    loss_2 = tf.reduce_mean(tf.abs(y_mixed - y_output[2]) , name='loss2')
    y_output.append(tf.multiply(x_mixed,y_pred[3]))
    loss_3 = tf.reduce_mean(tf.abs(y_mixed - y_output[3]) , name='loss3')
    loss_fn = loss_0+loss_1+loss_2+loss_3

    # Loss, Optimizer
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    learning_rate = tf.train.exponential_decay(NetConfig_MIR_1K.LR, global_step,
                                           NetConfig_MIR_1K.DECAY_STEP, NetConfig_MIR_1K.DECAY_RATE, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_fn, global_step=global_step)

x_input = np.zeros((batchSize, 512, 64, 1),dtype=np.float32)
y_input = np.zeros((batchSize, 512, 64, 2),dtype=np.float32)
displayIter = 100
lossAcc = 0
randperm = np.random.permutation(trainNum)
curIndex = 0
with tf.Session(config=NetConfig_MIR_1K.session_conf) as sess:

    # Initialized, Load state
    sess.run(tf.global_variables_initializer())

    for step in range(global_step.eval(), NetConfig_MIR_1K.FINAL_STEP):

        for i in range(batchSize):
            seq = randperm[curIndex]
            start = randint(0,trainMixed[seq].shape[-1]-64)
            x_input[i,:,:,:] = np.expand_dims(trainMixed[seq][0:512,start:start+64],2)
            y_input[i,:,:,0] = trainSrc1[seq][0:512,start:start+64]
            y_input[i,:,:,1] = trainSrc2[seq][0:512,start:start+64]
            curIndex = curIndex+1
            if curIndex == trainNum:
                curIndex = 0
                randperm = np.random.permutation(trainNum)

        l = sess.run([loss_fn, optimizer],
                                 feed_dict={x_mixed: x_input, y_mixed: y_input})

        lossAcc = lossAcc+l[0]
        if step%displayIter==0:
            print('step-{}\tloss={}'.format(step, lossAcc/displayIter))
            lossAcc = 0

        # Save state
        if step % NetConfig_MIR_1K.CKPT_STEP == 0:
            tf.train.Saver().save(sess, NetConfig_MIR_1K.CKPT_PATH + '/checkpoint', global_step=step)
