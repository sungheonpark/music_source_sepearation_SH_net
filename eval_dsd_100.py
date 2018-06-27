import numpy as np
import tensorflow as tf
from os import walk
import os
from model import infer
from config import NetConfig_DSD_100, ModelConfig
from util import to_spec, to_wav_file, bss_eval_sdr
import librosa
from statistics import median

batchSize = 1

# Model
print('Initialize network model')
with tf.device('/device:GPU:0'):
    x_mixed = tf.placeholder(tf.float32, shape=(batchSize, 512, 64, 1), name='x_mixed')
    y_mixed = tf.placeholder(tf.float32, shape=(batchSize, 512, 64, 2), name='y_mixed')
    y_pred = infer(x_mixed,4)
    #y_output = tf.multiply(x_mixed,y_pred)
    net = tf.make_template('net',y_pred)

x_input = np.zeros((batchSize, 512, 64, 1),dtype=np.float32)
#y_input = np.zeros((batchSize, 512, 64, 2),dtype=np.float32)

sdr_vocal = []
sdr_other = []
sdr_bass = []
sdr_drum = []

with tf.Session(config=NetConfig_DSD_100.session_conf) as sess:

    # Initialized, Load state
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(NetConfig_DSD_100.CKPT_PATH + '/checkpoint-150000'))
    if ckpt and ckpt.model_checkpoint_path:
        print('Load weights')
        tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)

        for (root, dirs, files) in walk(NetConfig_DSD_100.DATA_PATH+'/Mixtures/Test/'):
            for d in dirs:
                print(d)

                filenameBass = NetConfig_DSD_100.DATA_PATH+'/Sources/Test/'+d+'/bass.wav'
                filenameDrums = NetConfig_DSD_100.DATA_PATH+'/Sources/Test/'+d+'/drums.wav'
                filenameVocals = NetConfig_DSD_100.DATA_PATH+'/Sources/Test/'+d+'/vocals.wav'
                filenameOther = NetConfig_DSD_100.DATA_PATH+'/Sources/Test/'+d+'/other.wav'
                filenameMix = NetConfig_DSD_100.DATA_PATH+'/Mixtures/Test/'+d+'/mixture.wav'

                mixed_wav  = librosa.load(filenameMix, sr=ModelConfig.SR, mono=True)[0]
                gt_wav_bass = librosa.load(filenameBass, sr=ModelConfig.SR, mono=True)[0]
                gt_wav_drum = librosa.load(filenameDrums, sr=ModelConfig.SR, mono=True)[0]
                gt_wav_other = librosa.load(filenameOther, sr=ModelConfig.SR, mono=True)[0]
                gt_wav_vocal = librosa.load(filenameVocals, sr=ModelConfig.SR, mono=True)[0]
                mixed_spec = to_spec(mixed_wav)
                mixed_spec_mag = np.abs(mixed_spec)
                mixed_spec_phase = np.angle(mixed_spec)
                maxTemp = np.max(mixed_spec_mag)
                mixed_spec_mag = mixed_spec_mag/maxTemp

                mixed_wav_orig,sr_orig = librosa.load(filenameMix, sr=None, mono=True)
                gt_wav_bass_orig = librosa.load(filenameBass, sr=None, mono=True)[0]
                gt_wav_drum_orig = librosa.load(filenameDrums, sr=None, mono=True)[0]
                gt_wav_other_orig = librosa.load(filenameOther, sr=None, mono=True)[0]
                gt_wav_vocal_orig = librosa.load(filenameVocals, sr=None, mono=True)[0]

                srcLen = mixed_spec_mag.shape[-1]
                startIndex = 0
                y_est_bass = np.zeros((512,srcLen),dtype=np.float32)
                y_est_drum = np.zeros((512,srcLen),dtype=np.float32)
                y_est_other = np.zeros((512,srcLen),dtype=np.float32)
                y_est_vocal = np.zeros((512,srcLen),dtype=np.float32)
                while startIndex+64<srcLen:
                    x_input[0,:,:,0] = mixed_spec_mag[0:512,startIndex:startIndex+64]
                    y_output = sess.run(y_pred, feed_dict={x_mixed: x_input})
                    y_output = y_output[-1]
                    if startIndex==0:
                        y_est_bass[:,startIndex:startIndex+64] = y_output[0,:,:,0]
                        y_est_drum[:,startIndex:startIndex+64] = y_output[0,:,:,1]
                        y_est_other[:,startIndex:startIndex+64] = y_output[0,:,:,2]
                        y_est_vocal[:,startIndex:startIndex+64] = y_output[0,:,:,3]
                    else:
                        y_est_bass[:,startIndex+16:startIndex+48] = y_output[0,:,16:48,0]
                        y_est_drum[:,startIndex+16:startIndex+48] = y_output[0,:,16:48,1]
                        y_est_other[:,startIndex+16:startIndex+48] = y_output[0,:,16:48,2]
                        y_est_vocal[:,startIndex+16:startIndex+48] = y_output[0,:,16:48,3]
                    startIndex = startIndex+32

                x_input[0,:,:,0] = mixed_spec_mag[0:512,srcLen-64:srcLen]
                y_output = sess.run(y_pred, feed_dict={x_mixed: x_input})
                y_output = y_output[-1]
                srcStart = srcLen-startIndex-16
                y_est_bass[:,startIndex+16:srcLen] = y_output[0,:,64-srcStart:64,0]
                y_est_drum[:,startIndex+16:srcLen] = y_output[0,:,64-srcStart:64,1]
                y_est_other[:,startIndex+16:srcLen] = y_output[0,:,64-srcStart:64,2]
                y_est_vocal[:,startIndex+16:srcLen] = y_output[0,:,64-srcStart:64,3]

                y_est_bass[np.where(y_est_bass<0)] = 0
                y_est_drum[np.where(y_est_drum<0)] = 0
                y_est_other[np.where(y_est_other<0)] = 0
                y_est_vocal[np.where(y_est_vocal<0)] = 0
                y_est_bass = y_est_bass * mixed_spec_mag[0:512,:] * maxTemp
                y_est_drum = y_est_drum * mixed_spec_mag[0:512,:] * maxTemp
                y_est_other = y_est_other * mixed_spec_mag[0:512,:] * maxTemp
                y_est_vocal = y_est_vocal * mixed_spec_mag[0:512,:] * maxTemp
                y_wav_bass = to_wav_file(y_est_bass,mixed_spec_phase[0:512,:])
                y_wav_drum = to_wav_file(y_est_drum,mixed_spec_phase[0:512,:])
                y_wav_other = to_wav_file(y_est_other,mixed_spec_phase[0:512,:])
                y_wav_vocal = to_wav_file(y_est_vocal,mixed_spec_phase[0:512,:])


                #upsample to original SR
                y_wav_bass_orig = librosa.resample(y_wav_bass,ModelConfig.SR,sr_orig)
                y_wav_drum_orig = librosa.resample(y_wav_drum,ModelConfig.SR,sr_orig)
                y_wav_other_orig = librosa.resample(y_wav_other,ModelConfig.SR,sr_orig)
                y_wav_vocal_orig = librosa.resample(y_wav_vocal,ModelConfig.SR,sr_orig)

                sdr = bss_eval_sdr(gt_wav_bass_orig,y_wav_bass_orig)
                printstr = "Bass SDR : "
                printstr = printstr+str(sdr)+" Drum SDR : "
                sdr_bass.append(sdr)
                sdr = bss_eval_sdr(gt_wav_drum_orig,y_wav_drum_orig)
                printstr = printstr+str(sdr)+" Other SDR : "
                sdr_drum.append(sdr)
                sdr = bss_eval_sdr(gt_wav_other_orig,y_wav_other_orig)
                printstr = printstr+str(sdr)+" Vocal SDR : "
                sdr_other.append(sdr)
                sdr = bss_eval_sdr(gt_wav_vocal_orig,y_wav_vocal_orig)
                printstr = printstr+str(sdr)+" "
                sdr_vocal.append(sdr)
                print(printstr)

print('Median SDR')
print('Bass : ' + str(median(sdr_bass)) + ' Drum : ' + str(median(sdr_drum)) + ' Other : ' + str(median(sdr_other)) + ' Vocal : ' + str(median(sdr_vocal)))
