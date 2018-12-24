import torch
from torch.utils.data import Dataset

from .dataset import TwoFrameVideoSampler, final_image_transform
from .model import initialize_model
from .params import *


class PsuedoDataset(Dataset):
    def __init__(self, sampler):
        self.sampler = sampler
        self._open()

    def _open(self):
        self._iter = iter(self.sampler)

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        try:
            return next(self._iter)
        except StopIteration:
            self._open()
            return next(self._iter)


def predict_speeds(video_path):
    video_sampler = PsuedoDataset(TwoFrameVideoSampler(video_path, transformer=final_image_transform))
    sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(video_sampler), batch_size=batch_size, drop_last=True)
    batched_video = torch.utils.data.DataLoader(video_sampler, batch_sampler=sampler)

    model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    best_model_wts = torch.load('./best_model.pth')
    model.load_state_dict(best_model_wts)
    model.to(device)
    model.eval()

    for batch in batched_video:
        last_frame, current_frame = batch
        last_frame.to(device)
        current_frame.to(device)
        _speed = model(last_frame, current_frame)
        speed = _speed.tolist()
        print(speed)


def run_validation_routine():
    predict_speeds('./data/test.mp4')
