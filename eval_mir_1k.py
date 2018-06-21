import numpy as np
import tensorflow as tf
from os import walk
import os
from model import infer
from config import NetConfig_MIR_1K
from config import ModelConfig
from util import get_wav, to_spec, to_wav_file, bss_eval
import librosa

batchSize = 1

# Model
print('Initialize network model')
with tf.device('/device:GPU:0'):
    x_mixed = tf.placeholder(tf.float32, shape=(batchSize, 512, 64, 1), name='x_mixed')
    y_mixed = tf.placeholder(tf.float32, shape=(batchSize, 512, 64, 2), name='y_mixed')
    y_pred = infer(x_mixed,2)
    net = tf.make_template('net',y_pred)

x_input = np.zeros((batchSize, 512, 64, 1),dtype=np.float32)

gnsdr = 0.
gsir = 0.
gsar = 0.
totalLen = 0.

with tf.Session(config=NetConfig_MIR_1K.session_conf) as sess:

    # Initialized, Load state
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(NetConfig_MIR_1K.CKPT_PATH + '/checkpoint-15000'))
    if ckpt and ckpt.model_checkpoint_path:
        print('Load weights')
        tf.train.Saver().restore(sess, NetConfig_MIR_1K.CKPT_PATH + '/checkpoint-15000')

        for (root, dirs, files) in walk(NetConfig_MIR_1K.DATA_PATH):
            for f in files:
                if not (f.startswith("abjones") or f.startswith("amy")):
                    testFileName = '{}/{}'.format(NetConfig_MIR_1K.DATA_PATH,f)
                    [src1_src2_orig, sr_orig] = librosa.load(testFileName, sr=None, mono=False)
                    mixed_orig = librosa.to_mono(src1_src2_orig)
                    src1_orig, src2_orig = src1_src2_orig[0, :], src1_src2_orig[1, :]
                    mixed_wav, src1_wav, src2_wav = get_wav(testFileName)
                    mixed_spec = to_spec(mixed_wav)
                    src1_spec = to_spec(src1_wav)
                    src2_spec = to_spec(src2_wav)
                    mixed_spec_mag = np.abs(mixed_spec)
                    src1_spec_mag = np.abs(src1_spec)
                    src2_spec_mag = np.abs(src2_spec)
                    mixed_spec_phase = np.angle(mixed_spec)
                    maxTemp = np.max(mixed_spec_mag)
                    mixed_spec_mag = mixed_spec_mag/maxTemp

                    srcLen = mixed_spec_mag.shape[-1]
                    startIndex = 0
                    y_est_src1 = np.zeros((512,srcLen),dtype=np.float32)
                    y_est_src2 = np.zeros((512,srcLen),dtype=np.float32)

                    while startIndex+64<srcLen:
                        x_input[0,:,:,0] = mixed_spec_mag[0:512,startIndex:startIndex+64]
                        y_output = sess.run(y_pred, feed_dict={x_mixed: x_input})
                        y_output = y_output[-1]
                        if startIndex==0:
                            y_est_src1[:,startIndex:startIndex+64] = y_output[0,:,:,0]
                            y_est_src2[:,startIndex:startIndex+64] = y_output[0,:,:,1]
                        else:
                            y_est_src1[:,startIndex+16:startIndex+48] = y_output[0,:,16:48,0]
                            y_est_src2[:,startIndex+16:startIndex+48] = y_output[0,:,16:48,1]
                        startIndex = startIndex+32

                    x_input[0,:,:,0] = mixed_spec_mag[0:512,srcLen-64:srcLen]
                    y_output = sess.run(y_pred, feed_dict={x_mixed: x_input})
                    y_output = y_output[-1]
                    srcStart = srcLen-startIndex-16
                    y_est_src1[:,startIndex+16:srcLen] = y_output[0,:,64-srcStart:64,0]
                    y_est_src2[:,startIndex+16:srcLen] = y_output[0,:,64-srcStart:64,1]

                    y_est_src1[np.where(y_est_src1<0)] = 0
                    y_est_src2[np.where(y_est_src2<0)] = 0

                    y_est_src1 = y_est_src1 * mixed_spec_mag[0:512,:] * maxTemp
                    y_est_src2 = y_est_src2 * mixed_spec_mag[0:512,:] * maxTemp
                    y_wav1 = to_wav_file(y_est_src1,mixed_spec_phase[0:512,:])
                    y_wav2 = to_wav_file(y_est_src2,mixed_spec_phase[0:512,:])

                    #upsample to original SR
                    y_wav1_orig = librosa.resample(y_wav1,ModelConfig.SR,sr_orig)
                    y_wav2_orig = librosa.resample(y_wav2,ModelConfig.SR,sr_orig)
                    nsdr, sir, sar, lens = bss_eval(mixed_orig, src1_orig, src2_orig, y_wav1_orig, y_wav2_orig)

                    printstr = f + ' ' + str(nsdr) + ' ' + str(sir) + ' ' + str(sar)
                    print(printstr)

                    totalLen = totalLen+lens
                    gnsdr = gnsdr + nsdr * lens
                    gsir = gsir + sir * lens
                    gsar = gsar + sar * lens
print('Final results')
#print(totalLen)
print('GNSDR [Accompaniments, voice]')
print(gnsdr/totalLen)
print('GSIR [Accompaniments, voice]')
print(gsir/totalLen)
print('GSAR [Accompaniments, voice]')
print(gsar/totalLen)
