
import tensorflow as tf
from os import walk
import os
import numpy as np
import librosa
from util import to_spec
from model import infer
from config import NetConfig_DSD_100, ModelConfig
from random import *

#load data
trainMixed = []
trainBass = []
trainDrum = []
trainOther = []
trainVocal = []
trainNum = 0
batchSize = 4

print('generate train spectrograms')

for (root, dirs, files) in walk(NetConfig_DSD_100.DATA_PATH+'/Mixtures/Dev/'):
    for d in dirs:
        print(d)
        filenameBass = NetConfig_DSD_100.DATA_PATH+'/Sources/Dev/'+d+'/bass.wav'
        filenameDrums = NetConfig_DSD_100.DATA_PATH+'/Sources/Dev/'+d+'/drums.wav'
        filenameVocals = NetConfig_DSD_100.DATA_PATH+'/Sources/Dev/'+d+'/vocals.wav'
        filenameOther = NetConfig_DSD_100.DATA_PATH+'/Sources/Dev/'+d+'/other.wav'
        filenameMix = NetConfig_DSD_100.DATA_PATH+'/Mixtures/Dev/'+d+'/mixture.wav'
        mixed_wav = librosa.load(filenameMix, sr=ModelConfig.SR, mono=True)[0]
        bass_wav = librosa.load(filenameBass, sr=ModelConfig.SR, mono=True)[0]
        drums_wav = librosa.load(filenameDrums, sr=ModelConfig.SR, mono=True)[0]
        vocals_wav = librosa.load(filenameVocals, sr=ModelConfig.SR, mono=True)[0]
        other_wav = librosa.load(filenameOther, sr=ModelConfig.SR, mono=True)[0]
        mixed_spec = to_spec(mixed_wav)
        mixed_spec_mag = np.abs(mixed_spec)
        bass_spec = to_spec(bass_wav)
        bass_spec_mag = np.abs(bass_spec)
        drums_spec = to_spec(drums_wav)
        drums_spec_mag = np.abs(drums_spec)
        vocals_spec = to_spec(vocals_wav)
        vocals_spec_mag = np.abs(vocals_spec)
        other_spec = to_spec(other_wav)
        other_spec_mag = np.abs(other_spec)
        maxVal = np.max(mixed_spec_mag)

        trainMixed.append(mixed_spec_mag/maxVal)
        trainBass.append(bass_spec_mag/maxVal)
        trainDrum.append(drums_spec_mag/maxVal)
        trainVocal.append(vocals_spec_mag/maxVal)
        trainOther.append(other_spec_mag/maxVal)

        trainNum = trainNum+1

print('Number of training examples : {}'.format(trainNum))

# Model
print('Initialize network')
with tf.device('/device:GPU:0'):
    y_output=[]
    x_mixed = tf.placeholder(tf.float32, shape=(batchSize, 512, 64, 1), name='x_mixed')
    y_mixed = tf.placeholder(tf.float32, shape=(batchSize, 512, 64, 4), name='y_mixed')
    y_pred = infer(x_mixed,4)
    #net = tf.make_template('net',y_pred)
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
    learning_rate = tf.train.exponential_decay(NetConfig_DSD_100.LR, global_step,
                                           NetConfig_DSD_100.DECAY_STEP, NetConfig_DSD_100.DECAY_RATE, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_fn, global_step=global_step)

x_input = np.zeros((batchSize, 512, 64, 1),dtype=np.float32)
y_input = np.zeros((batchSize, 512, 64, 4),dtype=np.float32)

displayIter = 500
lossAcc = 0
randperm = np.random.permutation(trainNum)
curIndex = 0
with tf.Session(config=NetConfig_DSD_100.session_conf) as sess:

    # Initialized, Load state
    sess.run(tf.global_variables_initializer())

    for step in range(global_step.eval(), NetConfig_DSD_100.FINAL_STEP):

        for i in range(batchSize):
            seq = randperm[curIndex]
            start = randint(0,trainMixed[seq].shape[-1]-64)
            x_input[i,:,:,:] = np.expand_dims(trainMixed[seq][0:512,start:start+64],2)
            y_input[i,:,:,0] = trainBass[seq][0:512,start:start+64]
            y_input[i,:,:,1] = trainDrum[seq][0:512,start:start+64]
            y_input[i,:,:,2] = trainOther[seq][0:512,start:start+64]
            y_input[i,:,:,3] = trainVocal[seq][0:512,start:start+64]
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
        if step % NetConfig_DSD_100.CKPT_STEP == 0:
            tf.train.Saver().save(sess, NetConfig_DSD_100.CKPT_PATH + '/checkpoint', global_step=step)
