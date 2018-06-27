import tensorflow as tf

class ModelConfig:
    SR = 8000
    L_FRAME = 1024
    L_HOP = 256

class NetConfig_MIR_1K:
    NAME = '4stack_256_mir_1k'
    DATA_PATH = '../MIR-1K/Wavfile'
    CKPT_PATH = 'checkpoints/' + NAME
    LR = 0.0001
    DECAY_RATE = 0.2
    DECAY_STEP = 12000
    FINAL_STEP = 15001
    CKPT_STEP = 5000
    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 2},
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=0.95
        ),
        allow_soft_placement = True,
    )

class NetConfig_DSD_100:
    NAME = '4stack_256_dsd_100'
    DATA_PATH = '../DSD100'
    CKPT_PATH = 'checkpoints/' + NAME
    LR = 0.0001
    DECAY_RATE = 0.2
    DECAY_STEP = 120000
    FINAL_STEP = 150001
    CKPT_STEP = 50000
    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 2},
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=0.95
        ),
        allow_soft_placement = True,
    )
