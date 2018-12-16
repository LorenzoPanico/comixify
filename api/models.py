import os
import uuid

import cv2
import pafy
# from django.conf import settings
# from django.core.files import File
# from django.db import models
from settings.settings import config as settings

# from api.exceptions import TooLargeFile
from comic_layout.comic_layout import LayoutGenerator
from keyframes.keyframes import KeyFramesExtractor
from style_transfer.style_transfer import StyleTransfer
from utils import jj, profile

class DummyObject(object):
    """docstring for DummyObject."""
    def __init__(self):
        super(DummyObject, self).__init__()


class Video(object):
    def __init__(self, path):
      self.file = DummyObject()
      self.file.path = path

    @profile
    def download_from_youtube(self, yt_url):
        yt_pafy = pafy.new(yt_url)

        # Use the biggest possible quality with file size < MAX_FILE_SIZE and resolution <= 480px
        for stream in reversed(yt_pafy.videostreams):
            if stream.get_filesize() < settings.MAX_FILE_SIZE and int(stream.quality.split("x")[1]) <= 480:
                tmp_name = uuid.uuid4().hex + ".mp4"
                relative_path = jj('raw_videos', tmp_name)
                full_path = jj(settings.MEDIA_ROOT, relative_path)
                stream.download(full_path)
                self.file.name = relative_path
                break
        else:
            raise TooLargeFile()

    def create_comix(self, yt_url='', frames_mode=0, rl_mode=0, image_assessment_mode=0, style_transfer_mode=0):
        (keyframes, keyframes_timings), keyframes_extraction_time = KeyFramesExtractor.get_keyframes(
            video=self,
            frames_mode=frames_mode,
            rl_mode=rl_mode,
            image_assessment_mode=image_assessment_mode
        )
        stylized_keyframes, stylization_time = StyleTransfer.get_stylized_frames(frames=keyframes,
                                                                                 style_transfer_mode=style_transfer_mode)
        comic_image, layout_generation_time = LayoutGenerator.get_layout(frames=stylized_keyframes)

        timings = {
            'keyframes_extraction_time': keyframes_extraction_time,
            'stylization_time': stylization_time,
            'layout_generation_time': layout_generation_time,
            'keyframes_extraction_time_details': keyframes_timings
        }

        return comic, timings
