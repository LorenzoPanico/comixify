import os
import inspect

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


PERMITTED_VIDEO_EXTENSIONS = ['mp4', 'avi']
MAX_FILE_SIZE = 50000000
NUMBERS_OF_FRAMES_TO_SHOW = 10
TMP_DIR = 'tmp/'
GPU = True

FEATURE_BATCH_SIZE = 32
DEFAULT_FRAMES_SAMPLING_MODE = 0
DEFAULT_RL_MODE = 0
DEFAULT_IMAGE_ASSESSMENT_MODE = 0

DEFAULT_STYLE_TRANSFER_MODE = 0
COMIX_GAN_MODEL_PATH = os.path.join(BASE_DIR, 'ComixGAN', 'pretrained_models', 'generator_model.h5')
MAX_FRAME_SIZE_FOR_STYLE_TRANSFER = 600

NIMA_MODEL_PATH = os.path.join(BASE_DIR, 'neural_image_assessment', 'pretrained_model', 'nima_model.h5')

# CAFFE_ROOT = 'caffe_git/'

# keys = ['BASE_DIR', 'PERMITTED_VIDEO_EXTENSIONS', 'MAX_FILE_SIZE', 'NUMBERS_OF_FRAMES_TO_SHOW',
#         'GPU', 'TMP_DIR', 'FEATURE_BATCH_SIZE', 'DEFAULT_FRAMES_SAMPLING_MODE',
#         'DEFAULT_RL_MODE', 'DEFAULT_IMAGE_ASSESSMENT_MODE', 'DEFAULT_STYLE_TRANSFER_MODE',
#         'COMIX_GAN_MODEL_PATH', 'MAX_FRAME_SIZE_FOR_STYLE_TRANSFER', 'NIMA_MODEL_PATH']


config_dict = {name: value for (name, value) in locals().items() if not name.startswith('_')}

class SettingObject(object):
    def __init__(self):
        super(SettingObject, self).__init__()

settings = SettingObject()

for key, value in config_dict.items():
    if not inspect.ismodule(value):
        # if not isinstance(value, list):
        #     os.environ[key] = str(value)
        setattr(settings, key, value)
