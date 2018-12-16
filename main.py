import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = '/content/comixify/'

CAFFE_ROOT = '/content/caffe/'

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.0/howto/static-files/

STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'static')

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

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


import errno
import os

import tensorflow as tf
from keras.models import load_model
from keras_contrib.layers import InstanceNormalization


class ComixGAN:
    def __init__(self):
        if not os.path.exists(COMIX_GAN_MODEL_PATH):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), COMIX_GAN_MODEL_PATH)
        self.graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.30
        config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            with self.session.as_default():
                with tf.device('/device:GPU:0'):
                    self.model = load_model(COMIX_GAN_MODEL_PATH,
                                            custom_objects={'InstanceNormalization': InstanceNormalization})

import cv2
import numpy as np



class LayoutGenerator():
    @classmethod
    def get_layout(cls, frames):
        result_imgs = cls._pad_images(frames)

        first_row = np.hstack(result_imgs[:2])
        second_row = np.hstack(result_imgs[2:5])
        third_row = np.hstack(result_imgs[5:7])
        fourth_row = np.hstack(result_imgs[7:10])

        second_row = cv2.resize(second_row,
                                (first_row.shape[1],
                                 (second_row.shape[0] * first_row.shape[1]) // second_row.shape[1]))
        fourth_row = cv2.resize(fourth_row,
                                (first_row.shape[1],
                                 (fourth_row.shape[0] * first_row.shape[1]) // fourth_row.shape[1]))

        return np.vstack([first_row, second_row, third_row, fourth_row])

    @staticmethod
    def _pad_images(frames):
        padded_result_imgs = []
        for img in frames:
            padded_img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            padded_result_imgs.append(padded_img)
        return padded_result_imgs

import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from CartoonGAN.network.Transformer import Transformer

# load pretrained model
comixGAN = ComixGAN()


class StyleTransfer():
    @classmethod
    def get_stylized_frames(cls, frames, style_transfer_mode=0, gpu=GPU):
        if style_transfer_mode == 0:
            return cls._comix_gan_stylize(frames=frames)
        elif style_transfer_mode == 1:
            return cls._cartoon_gan_stylize(frames, gpu=gpu, style='Hayao')
        elif style_transfer_mode == 2:
            return cls._cartoon_gan_stylize(frames, gpu=gpu, style='Hosoda')

    @staticmethod
    def _resize_images(frames, size=384):
        resized_images = []
        for img in frames:
            # resize image, keep aspect ratio
            h, w, _ = img.shape
            ratio = h / w
            if ratio > 1:
                h = size
                w = int(h * 1.0 / ratio)
            else:
                w = size
                h = int(w * ratio)
            resized_img = cv2.resize(img, (w, h))
            resized_images.append(resized_img)
        return resized_images

    @classmethod
    def _comix_gan_stylize(cls, frames):
        if max(frames[0].shape) > MAX_FRAME_SIZE_FOR_STYLE_TRANSFER:
            frames = cls._resize_images(frames, size=MAX_FRAME_SIZE_FOR_STYLE_TRANSFER)

        with comixGAN.graph.as_default():
            with comixGAN.session.as_default():
                batch_size = 2
                stylized_imgs = []
                for i in range(0, len(frames), batch_size):
                    batch_of_frames = ((np.stack(frames[i:i + batch_size]) / 255) * 2) - 1
                    stylized_batch_of_imgs = comixGAN.model.predict(batch_of_frames)
                    stylized_imgs.append(255 * ((stylized_batch_of_imgs + 1) / 1.25))

        return list(np.concatenate(stylized_imgs, axis=0))

    @classmethod
    def _cartoon_gan_stylize(cls, frames, gpu=True, style='Hayao'):
        model = None

        if model is None:
            # load pretrained model
            model = Transformer()
            model.load_state_dict(torch.load(os.path.join("CartoonGAN/pretrained_model", style + "_net_G_float.pth")))
            model.eval()
            model.cuda() if gpu else model.float()

        frames = cls._resize_images(frames, size=450)
        stylized_imgs = []
        for img in frames:
            input_image = transforms.ToTensor()(img).unsqueeze(0)

            # preprocess, (-1, 1)
            input_image = -1 + 2 * input_image
            input_image = Variable(input_image).cuda() if gpu else Variable(input_image).float()

            # forward
            output_image = model(input_image)
            output_image = output_image[0]

            # deprocess, (0, 1)
            output_image = (output_image.data.cpu().float() * 0.5 + 0.5).numpy()

            # switch channels -> (c, h, w) -> (h, w, c)
            output_image = np.rollaxis(output_image, 0, 3)

            # append image to result images
            stylized_imgs.append(255 * output_image)

        return stylized_imgs

import os
import uuid

import cv2
import pafy

from utils import jj


class DummyObject(object):
    """docstring for DummyObject."""
    def __init__(self):
        super(DummyObject, self).__init__()


class Video(object):
    def __init__(self, path):
      self.file = DummyObject()
      self.file.path = path

    def download_from_youtube(self, yt_url):
        yt_pafy = pafy.new(yt_url)
        self.file = DummyObject()
        # Use the biggest possible quality with file size < MAX_FILE_SIZE and resolution <= 480px
        for stream in reversed(yt_pafy.videostreams):
            if stream.get_filesize() < MAX_FILE_SIZE and int(stream.quality.split("x")[1]) <= 480:
                tmp_name = uuid.uuid4().hex + ".mp4"
                relative_path = jj('raw_videos', tmp_name)
                full_path = jj(MEDIA_ROOT, relative_path)
                stream.download(full_path)
                self.file.name = relative_path
                break
        else:
            raise TooLargeFile()

    def create_comix(self, yt_url='', frames_mode=0, rl_mode=0, image_assessment_mode=0, style_transfer_mode=0,
                     keyframes=None):

        if keyframes is None:
          keyframes = KeyFramesExtractor.get_keyframes(
              video=self,
              frames_mode=frames_mode,
              rl_mode=rl_mode,
              image_assessment_mode=image_assessment_mode
          )

        if style_transfer_mode < 0:
          print("Skipping style transfer")
          stylized_keyframes = keyframes
        else:
          stylized_keyframes = StyleTransfer.get_stylized_frames(frames=keyframes, style_transfer_mode=style_transfer_mode)

        assert stylized_keyframes is not None

        comic_image = LayoutGenerator.get_layout(frames=stylized_keyframes)


        return comic_image

import errno
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from keras.applications.nasnet import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import img_to_array


class NeuralImageAssessment:
    def __init__(self):
        if not os.path.exists(NIMA_MODEL_PATH):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), NIMA_MODEL_PATH)
        self.graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            with self.session.as_default():
                self.model = load_model(NIMA_MODEL_PATH)

    @staticmethod
    def resize_image(bgr_img_array, target_size=(224, 224), interpolation='nearest'):
        _PIL_INTERPOLATION_METHODS = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
        }

        img = Image.fromarray(np.uint8(bgr_img_array[..., ::-1]))
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
        return img

    def get_assessment_score(self, img_array):
        with self.graph.as_default():
            with self.session.as_default():
                target_size = (224, 224)
                img = NeuralImageAssessment.resize_image(img_array, target_size)
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                scores = self.model.predict(x, batch_size=1, verbose=0)[0]
        mean = NeuralImageAssessment.mean_score(scores)

        return mean

    @staticmethod
    def mean_score(scores):
        si = np.arange(1, 11, 1)
        mean = np.sum(scores * si)
        return mean

    @staticmethod
    def std_score(scores):
        si = np.arange(1, 11, 1)
        mean = NeuralImageAssessment.mean_score(scores)
        std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
        return std

import os
import uuid
import shutil
import numpy as np
import torch
import torch.nn as nn

os.environ['GLOG_minloglevel'] = '2'  # Prevent caffe shell loging
import caffe
from subprocess import call
from math import ceil
from sklearn.preprocessing import normalize
from skimage import img_as_ubyte
import logging

from utils import jj
from keyframes_rl.models import DSN
from popularity.models import PopularityPredictor
from keyframes.kts import cpd_auto
from keyframes.utils import batch

logger = logging.getLogger(__name__)

nima_model = NeuralImageAssessment()


class KeyFramesExtractor:
    @classmethod
    def get_keyframes(cls, video, gpu=GPU, features_batch_size=FEATURE_BATCH_SIZE,
                      frames_mode=0, rl_mode=0, image_assessment_mode=0):
        (frames_paths, all_frames_tmp_dir) = cls._get_all_frames(video, mode=frames_mode)
        frames = cls._get_frames(frames_paths)
        features = cls._get_features(frames, gpu, features_batch_size)
        norm_features = normalize(features)
        change_points, frames_per_segment = cls._get_segments(norm_features)
        probs = cls._get_probs(norm_features, gpu, mode=rl_mode)
        keyframes = cls._get_keyframes(frames, probs, change_points, frames_per_segment)
        chosen_frames = cls._get_popularity_chosen_frames(keyframes, features, image_assessment_mode)
        shutil.rmtree(jj(f"{TMP_DIR}", f"{all_frames_tmp_dir}"))
        return chosen_frames

    @staticmethod
    def _get_all_frames(video, mode=0):
        all_frames_tmp_dir = uuid.uuid4().hex
        os.mkdir(jj(TMP_DIR, all_frames_tmp_dir))
        if mode == 1:
            call(["ffmpeg", "-i", f"{video.file.path}", "-c:v", "libxvid", "-qscale:v", "1", "-an",
                  jj(f"{TMP_DIR}", f"{all_frames_tmp_dir}", "video.mp4")])
            call(["ffmpeg", "-i", jj(f"{TMP_DIR}", f"{all_frames_tmp_dir}", "video.mp4"), "-vf",
                  "select=eq(pict_type\,I)", "-vsync", "vfr",
                  jj(f"{TMP_DIR}", f"{all_frames_tmp_dir}", "%06d.jpeg")])
        else:
            call(["ffmpeg", "-i", video.file.path, "-vf", "select=not(mod(n\\,15))", "-vsync", "vfr", "-q:v", "2",
                  jj(TMP_DIR, all_frames_tmp_dir, "%06d.bmp")])
        frames_paths = []
        for dirname, dirnames, filenames in os.walk(jj(TMP_DIR, all_frames_tmp_dir)):
            for filename in filenames:
                if not filename.endswith(".mp4"):
                    frames_paths.append(jj(dirname, filename))
        return sorted(frames_paths), all_frames_tmp_dir

    @staticmethod
    def _get_frames(frames_paths):
        frames = []
        for frame_path in frames_paths:
            frame = caffe.io.load_image(frame_path)
            frames.append(frame)
        return frames

    @staticmethod
    def _get_features(frames, gpu=True, batch_size=1):
        caffe_root = CAFFE_ROOT
        if not caffe_root:
            print("Caffe root path not found.")
        if not gpu:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()

        model_file = caffe_root + "/models/bvlc_googlenet/deploy.prototxt"
        pretrained = caffe_root + "/models/bvlc_googlenet/bvlc_googlenet.caffemodel"
        if not os.path.isfile(pretrained):
            print("PRETRAINED Model not found.")

        net = caffe.Net(model_file, pretrained, caffe.TEST)
        net.blobs["data"].reshape(batch_size, 3, 224, 224)

        mu = np.load(caffe_root + "/python/caffe/imagenet/ilsvrc_2012_mean.npy")
        mu = mu.mean(1).mean(1)
        transformer = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
        transformer.set_transpose("data", (2, 0, 1))
        transformer.set_mean("data", mu)
        transformer.set_raw_scale("data", 255)
        transformer.set_channel_swap("data", (2, 1, 0))

        features = np.zeros(shape=(len(frames), 1024))
        for idx_batch, (n_batch, frames_batch) in enumerate(batch(frames, batch_size)):
            for i in range(n_batch):
                net.blobs['data'].data[i, ...] = transformer.preprocess("data", frames_batch[i])
            net.forward()
            temp = net.blobs["pool5/7x7_s1"].data[0:n_batch]
            temp = temp.squeeze().copy()
            features[idx_batch * batch_size:idx_batch * batch_size + n_batch] = temp
        return features.astype(np.float32)

    @staticmethod
    def _get_probs(features, gpu=True, mode=0):
        model = None

        if model is None:
            if mode == 1:
                model_path = "keyframes_rl/pretrained_model/model_1.pth.tar"
            else:
                model_path = "keyframes_rl/pretrained_model/model_0.pth.tar"
            model = DSN(in_dim=1024, hid_dim=256, num_layers=1, cell="lstm")
            if gpu:
                checkpoint = torch.load(model_path)
            else:
                checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint)
            if gpu:
                model = nn.DataParallel(model).cuda()
            model.eval()

        seq = torch.from_numpy(features).unsqueeze(0)
        if gpu: seq = seq.cuda()
        probs = model(seq)
        probs = probs.data.cpu().squeeze().numpy()
        return probs

    @staticmethod
    def _get_keyframes(frames, probs, change_points, frames_per_segment, min_keyframes=20):
        gts = []
        s = 0
        for q in frames_per_segment:
            gts.append(np.mean(probs[s:s + q]).astype(float))
            s += q
        gts = np.array(gts)
        picks = np.argsort(gts)[::-1][:min_keyframes]
        chosen_frames = []
        for pick in picks:
            cp = change_points[pick]
            low = cp[0]
            high = cp[1]
            x = low
            if low != high:
                x = low + np.argmax(probs[low:high])
            chosen_frames.append({
                "index": x,
                "frame": img_as_ubyte(frames[x])[..., ::-1]
            })
        chosen_frames.sort(key=lambda k: k['index'])
        return chosen_frames

    @staticmethod
    def _get_popularity_chosen_frames(frames, features, image_assessment_mode=0, n_frames=10):
        if image_assessment_mode == 1:
            model = None
            if model is None:
                model = PopularityPredictor()
            for frame in frames:
                x = features[frame["index"]]
                frame["popularity"] = model.get_popularity_score(x).squeeze()
        else:
            for frame in frames:
                x = frame["frame"]
                frame["popularity"] = nima_model.get_assessment_score(x)
        chosen_frames = []
        for frame_0, frame_1 in zip(frames[0::2], frames[1::2]):
            if frame_0["popularity"] > frame_1["popularity"]:
                chosen_frames.append(frame_0)
            else:
                chosen_frames.append(frame_1)
        return [o["frame"] for o in chosen_frames]

    @staticmethod
    def _get_segments(features):
        K = np.dot(features, features.T)
        n_frames = int(K.shape[0])
        min_segments = int(ceil(n_frames / 20))
        min_segments = max(20, min_segments)
        min_segments = min(n_frames - 1, min_segments)
        cps, scores = cpd_auto(K, min_segments, 1, min_segments=min_segments)
        change_points = [
            [0, cps[0] - 1]
        ]
        frames_per_segment = [int(cps[0])]
        for j in range(0, len(cps) - 1):
            change_points.append([cps[j], cps[j + 1] - 1])
            frames_per_segment.append(int(cps[j + 1] - cps[j]))
        frames_per_segment.append(int(len(features) - cps[len(cps) - 1]))
        change_points.append([cps[len(cps) - 1], len(features) - 1])
        print("Number of segments: " + str(len(frames_per_segment)))
        return change_points, frames_per_segment
