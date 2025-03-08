"""
Fixed conditional loggers for VQ-GAN and diffusion models
Compatible with PyTorch Lightning's SummaryWriter
"""

import os
import torch
import numpy as np
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
import torchvision
from PIL import Image

from train.callbacks import save_video_grid  # Import from your existing code


class ConditionalImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.log_steps = [
            2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        
        for k in images:
            images[k] = (images[k] + 1.0) * 127.5
            torch.clamp(images[k], 0, 255)
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                # Handle different batch formats to extract condition
                if isinstance(batch, dict) and 'condition' in batch:
                    # Dict format with condition
                    images = pl_module.log_images(batch)
                elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    if isinstance(batch[1], dict) and 'condition' in batch[1]:
                        # Tuple format with second element as dict containing condition
                        images = pl_module.log_images({
                            'data': batch[0],
                            'condition': batch[1]['condition']
                        })
                    else:
                        # Regular tuple format
                        images = pl_module.log_images(batch)
                else:
                    # Original format
                    images = pl_module.log_images(batch, split=split)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")


class ConditionalVideoLogger(Callback):
    def __init__(self, batch_frequency, max_videos, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_videos = max_videos
        self.log_steps = [
            2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def log_local(self, save_dir, split, videos,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "videos", split)
        
        for k in videos:
            videos[k] = (videos[k] + 1.0) * 127.5
            torch.clamp(videos[k], 0, 255)
            videos[k] = videos[k] / 255.0
            grid = videos[k]
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.mp4".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            save_video_grid(grid, path)

    def log_vid(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and
                hasattr(pl_module, "log_videos") and
                callable(pl_module.log_videos) and
                self.max_videos > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                # Handle different batch formats to extract condition
                if isinstance(batch, dict) and 'condition' in batch:
                    # Dict format with condition
                    videos = pl_module.log_videos(batch)
                elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    if isinstance(batch[1], dict) and 'condition' in batch[1]:
                        # Tuple format with second element as dict containing condition
                        videos = pl_module.log_videos({
                            'data': batch[0],
                            'condition': batch[1]['condition']
                        })
                    else:
                        # Regular tuple format
                        videos = pl_module.log_videos(batch)
                else:
                    # Original format
                    videos = pl_module.log_videos(batch, split=split, batch_idx=batch_idx)
                
                # Limit to first 2 keys as in original code
                keys = list(videos.keys())
                videos = {k: videos[k] for k in keys[:2]}

            for k in videos:
                N = min(videos[k].shape[0], self.max_videos)
                videos[k] = videos[k][:N]
                if isinstance(videos[k], torch.Tensor):
                    videos[k] = videos[k].detach().cpu()

            self.log_local(pl_module.logger.save_dir, split, videos,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_vid(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_vid(pl_module, batch, batch_idx, split="val")