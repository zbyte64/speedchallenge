import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import datasets, models, transforms


def read_video_frames(video_file):
    print(video_file)
    cap = cv2.VideoCapture(video_file)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    x_size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    y_size = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    channels = 3
    frames = torch.FloatTensor(channels, n_frames, x_size, y_size)
    failedClip = False
    for f in range(n_frames):
        ret, frame = cap.read()
        if ret:
            frame = torch.from_numpy(frame)
            # HWC2CHW
            frame = frame.permute(2, 0, 1)
            frames[:, f, :, :] = frame
        else:
            assert False

    #TODO normalize
    #mean = [0,0,0]
    #for c in range(3):
    #    frames[c] -= mean[c]
    frames /= 255
    return frames


class VideoSampler(Sampler):
    def __init__(self, path, transformer=None):
        self.vid_capture = _c = cv2.VideoCapture(path)
        self.n_frames = int(_c.get(cv2.CAP_PROP_FRAME_COUNT))
        self.x_size = int(_c.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.y_size = int(_c.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.transformer = transformer

    def __len__(self):
        return self.n_frames

    def __iter__(self):
        for f in range(self.n_frames):
            ret, frame = self.vid_capture.read()
            if ret:
                frame = Image.fromarray(frame)
                if self.transformer:
                    frame = self.transformer(frame)
                else:
                    frame = torch.from_numpy(frame.astype(np.float32))
                    # HWC2CHW
                    frame = frame.permute(2, 0, 1)
                    frame /= 255
                yield frame
            else:
                assert False


class SpeedVideoSampler(Dataset):
    """Sequential Dataset Class for Loading Video"""

    def __init__(self, path, max_speed=None, transformer=None):
        self.path = path
        video_file = os.path.join(path, 'train.mp4')
        speed_file = os.path.join(path, 'train.txt')
        self.frame_sampler = VideoSampler(video_file, transformer=transformer)
        print('#'*30)
        print(self.frame_sampler)
        self.speeds = torch.FloatTensor(list(map(float, open(speed_file, 'r').readlines())))
        self.max_speed = max_speed or max(self.speeds)
        self.transformer = transformer
        self.speeds /= self.max_speed
        self._frames = iter(self.frame_sampler)
        self._last_idx = None
        self._ds_iter = iter(self) #yikes, for use with sequential sampling only

    def __len__(self):
        return len(self.frame_sampler)-1

    def __iter__(self):
        last_frame = next(self._frames)
        for idx in range(len(self)):
            current_frame = next(self._frames)
            yield (last_frame, current_frame, self.speeds[idx])
            last_frame = current_frame

    def __getitem__(self, idx):
        if self._last_idx:
            assert self._last_idx == idx - 1, "Out of order read"
        self._last_idx = idx
        return next(self._ds_iter)
