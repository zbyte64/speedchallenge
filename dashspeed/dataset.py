import os
import cv2
import torch
import numpy as np
import random
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import datasets, models, transforms


final_image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class TwoFrameVideoSampler(Sampler):
    def __init__(self, path, transformer=transforms.ColorJitter()):
        self.vid_capture = _c = cv2.VideoCapture(path)
        self.n_frames = int(_c.get(cv2.CAP_PROP_FRAME_COUNT))
        self.x_size = int(_c.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.y_size = int(_c.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.transformer = transformer

    def __len__(self):
        return self.n_frames - 1

    def __iter__(self):
        last_frame = None
        for f in range(self.n_frames):
            ret, frame = self.vid_capture.read()
            if ret:
                frame = Image.fromarray(frame)
                frame = self.transformer(frame)
                if last_frame is not None:
                    yield last_frame, frame
                last_frame = frame
            else:
                assert False


class SpeedVideoSampler(Dataset):
    """
    Sequential Dataset Class for Loading Dashcam Video with speeds

    Randomly rotates/flips pairs of images
    """

    def __init__(self, path, max_speed=None):
        self.path = path
        speed_file = os.path.join(path, 'train.txt')
        self.speeds = torch.FloatTensor(list(map(float, open(speed_file, 'r').readlines())))
        self.max_speed = max_speed or max(self.speeds)
        self.speeds /= self.max_speed
        self._open()

    def _open(self):
        video_file = os.path.join(self.path, 'train.mp4')
        self.frame_sampler = TwoFrameVideoSampler(video_file)
        self._frames = iter(self.frame_sampler)
        self._last_idx = None
        self._ds_iter = iter(self) #yikes, for use with sequential sampling only

    def __len__(self):
        return len(self.frame_sampler)

    def __iter__(self):
        for idx in range(len(self)):
            last_frame, current_frame = next(self._frames)
            t = {
                0: lambda x: final_image_transform(x),
                1: lambda x: final_image_transform(x.transpose(PIL.Image.FLIP_LEFT_RIGHT)),
                2: lambda x: final_image_transform(x.transpose(PIL.Image.FLIP_TOP_BOTTOM)),
                3: lambda x: final_image_transform(x.transpose(PIL.Image.ROTATE_180)),
            }[random.randint(0, 3)]
            if random.randint(0, 1):
                yield (t(last_frame), t(current_frame), self.speeds[idx])
            else:
                #reverse time
                yield (t(current_frame), t(last_frame), -self.speeds[idx])

    def __getitem__(self, idx):
        if self._last_idx and self._last_idx != idx - 1:
            self._open()
        self._last_idx = idx
        return next(self._ds_iter)
