from dashspeed.train import run_training_routine
from dashspeed.label_video import run_validation_routine

import os, sys


if __name__ == '__main__':
    cmd = sys.argv[-1]
    if cmd == 'train':
        run_training_routine()
    if cmd == 'test':
        run_validation_routine()
