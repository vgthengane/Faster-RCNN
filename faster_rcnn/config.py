from easydict import EasyDict as edict

_config = edict()

cfg = _config

# Training options
# ----------------

_config.TRAIN = edict()
_config.TRAIN.ANCHOR_SIZES = [128, 256, 512]
_config.TRAIN.ASPECT_RATIOS = [0.5, 1.0, 2.0]
_config.TRAIN.OUT_CHANNELS = 512
_config.TRAIN.IMG_SIZE = (800, 800)
_config.TRAIN.RPN_PRE_NMS_TOP_N = 6000
_config.TRAIN.RPN_POST_NMS_TOP_N = 2000
_config.TRAIN.RPN_NMS_THRESH = 0.7
_config.TRAIN.RPN_MIN_SIZE = 32
_config.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
_config.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
_config.TRAIN.NUM_SAMPLES = 256
_config.TRAIN.POSITIVE_RATIO = 0.5


