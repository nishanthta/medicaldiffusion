"""Conditional VQ-GAN implementation for 3D MRI volumes
Adapted from the original VQ-GAN implementation with added conditioning on scan type
"""

import math
import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from vq_gan_3d.utils import shift_dim, adopt_weight, comp_getattr
from vq_gan_3d.model.lpips import LPIPS
from vq_gan_3d.model.codebook import Codebook


def silu(x):
    return x*torch.sigmoid(x)


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return silu(x)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class ConditionalVQGAN(pl.LightningModule):
    def __init__(self, cfg, num_conditions=3):
        """
        Conditional VQ-GAN for 3D MRI volumes
        
        Args:
            cfg: Configuration object
            num_conditions: Number of condition types (e.g., 3 for arterial, venous, delayed)
        """
        super().__init__()
        self.cfg = cfg
        self.embedding_dim = cfg.model.embedding_dim 
        self.n_codes = cfg.model.n_codes
        self.num_conditions = num_conditions
        
        # Condition embedding
        self.condition_embedding = nn.Embedding(num_conditions, cfg.model.condition_dim)
        
        # Main components
        self.encoder = ConditionalEncoder(
            cfg.model.n_hiddens, 
            cfg.model.downsample,
            cfg.dataset.image_channels, 
            cfg.model.norm_type, 
            cfg.model.padding_type,
            cfg.model.num_groups,
            cfg.model.condition_dim
        )
        
        self.decoder = ConditionalDecoder(
            cfg.model.n_hiddens, 
            cfg.model.downsample, 
            cfg.dataset.image_channels, 
            cfg.model.norm_type, 
            cfg.model.num_groups,
            cfg.model.condition_dim
        )
        
        self.enc_out_ch = self.encoder.out_channels
        self.pre_vq_conv = SamePadConv3d(
            self.enc_out_ch, cfg.model.embedding_dim, 1, padding_type=cfg.model.padding_type)
        self.post_vq_conv = SamePadConv3d(
            cfg.model.embedding_dim, self.enc_out_ch, 1)

        self.codebook = Codebook(cfg.model.n_codes, cfg.model.embedding_dim,
                                 no_random_restart=cfg.model.no_random_restart, restart_thres=cfg.model.restart_thres)

        self.gan_feat_weight = cfg.model.gan_feat_weight
        
        # Discriminators
        self.image_discriminator = ConditionalNLayerDiscriminator(
            cfg.dataset.image_channels, 
            cfg.model.disc_channels, 
            cfg.model.disc_layers, 
            norm_layer=nn.BatchNorm2d,
            condition_dim=cfg.model.condition_dim
        )
        
        self.video_discriminator = ConditionalNLayerDiscriminator3D(
            cfg.dataset.image_channels, 
            cfg.model.disc_channels, 
            cfg.model.disc_layers, 
            norm_layer=nn.BatchNorm3d,
            condition_dim=cfg.model.condition_dim
        )

        if cfg.model.disc_loss_type == 'vanilla':
            self.disc_loss = vanilla_d_loss
        elif cfg.model.disc_loss_type == 'hinge':
            self.disc_loss = hinge_d_loss

        self.perceptual_model = LPIPS().eval()

        self.image_gan_weight = cfg.model.image_gan_weight
        self.video_gan_weight = cfg.model.video_gan_weight

        self.perceptual_weight = cfg.model.perceptual_weight

        self.l1_weight = cfg.model.l1_weight
        self.save_hyperparameters()

    def encode(self, x, condition, include_embeddings=False, quantize=True):
        """
        Encode input with condition to latent representation
        
        Args:
            x: Input tensor [B, C, T, H, W]
            condition: Condition tensor [B]
            include_embeddings: Whether to include embeddings in output
            quantize: Whether to quantize the latent representation
            
        Returns:
            Encoded representation
        """
        condition_emb = self.condition_embedding(condition)
        h = self.encoder(x, condition_emb)
        h = self.pre_vq_conv(h)
        
        if quantize:
            vq_output = self.codebook(h)
            if include_embeddings:
                return vq_output['embeddings'], vq_output['encodings']
            else:
                return vq_output['encodings']
        return h

    def decode(self, latent, condition, quantize=False):
        """
        Decode latent representation with condition to image
        
        Args:
            latent: Latent representation
            condition: Condition tensor [B]
            quantize: Whether to quantize the latent representation
            
        Returns:
            Reconstructed image
        """
        condition_emb = self.condition_embedding(condition)
        
        if quantize:
            vq_output = self.codebook(latent)
            latent = vq_output['encodings']
            
        h = F.embedding(latent, self.codebook.embeddings)
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        return self.decoder(h, condition_emb)

    def forward(self, x, condition, optimizer_idx=None, log_image=False):
        """
        Forward pass with condition
        
        Args:
            x: Input tensor [B, C, T, H, W]
            condition: Condition tensor [B]
            optimizer_idx: Optimizer index for GAN training
            log_image: Whether to log images
            
        Returns:
            Model outputs based on optimizer_idx
        """
        B, C, T, H, W = x.shape
        condition_emb = self.condition_embedding(condition)

        # Encode and get VQ output
        z = self.encoder(x, condition_emb)
        z = self.pre_vq_conv(z)
        vq_output = self.codebook(z)
        
        # Decode
        x_recon = self.decoder(self.post_vq_conv(vq_output['embeddings']), condition_emb)

        # Calculate reconstruction loss
        recon_loss = F.l1_loss(x_recon, x) * self.l1_weight

        # Select random frames for 2D discriminator
        frame_idx = torch.randint(0, T, [B]).cuda()
        frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
        frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)
        frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)

        if log_image:
            return frames, frames_recon, x, x_recon

        if optimizer_idx == 0:
            # Autoencoder - train the "generator"
            # Perceptual loss
            perceptual_loss = 0
            if self.perceptual_weight > 0:
                perceptual_loss = self.perceptual_model(frames, frames_recon).mean() * self.perceptual_weight

            # Discriminator loss (turned on after a certain epoch)
            logits_image_fake, pred_image_fake = self.image_discriminator(frames_recon, condition_emb)
            logits_video_fake, pred_video_fake = self.video_discriminator(x_recon, condition_emb)
            
            g_image_loss = -torch.mean(logits_image_fake)
            g_video_loss = -torch.mean(logits_video_fake)
            g_loss = self.image_gan_weight*g_image_loss + self.video_gan_weight*g_video_loss
            
            disc_factor = adopt_weight(
                self.global_step, threshold=self.cfg.model.discriminator_iter_start)
            aeloss = disc_factor * g_loss

            # GAN feature matching loss
            image_gan_feat_loss = 0
            video_gan_feat_loss = 0
            feat_weights = 4.0 / (3 + 1)
            
            if self.image_gan_weight > 0:
                logits_image_real, pred_image_real = self.image_discriminator(frames, condition_emb)
                for i in range(len(pred_image_fake)-1):
                    image_gan_feat_loss += feat_weights * \
                        F.l1_loss(pred_image_fake[i], pred_image_real[i].detach()) * (self.image_gan_weight > 0)
            
            if self.video_gan_weight > 0:
                logits_video_real, pred_video_real = self.video_discriminator(x, condition_emb)
                for i in range(len(pred_video_fake)-1):
                    video_gan_feat_loss += feat_weights * \
                        F.l1_loss(pred_video_fake[i], pred_video_real[i].detach()) * (self.video_gan_weight > 0)
            
            gan_feat_loss = disc_factor * self.gan_feat_weight * (image_gan_feat_loss + video_gan_feat_loss)

            # Log metrics
            self.log("train/g_image_loss", g_image_loss, logger=True, on_step=True, on_epoch=True)
            self.log("train/g_video_loss", g_video_loss, logger=True, on_step=True, on_epoch=True)
            self.log("train/image_gan_feat_loss", image_gan_feat_loss, logger=True, on_step=True, on_epoch=True)
            self.log("train/video_gan_feat_loss", video_gan_feat_loss, logger=True, on_step=True, on_epoch=True)
            self.log("train/perceptual_loss", perceptual_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/recon_loss", recon_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/commitment_loss", vq_output['commitment_loss'], prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log('train/perplexity', vq_output['perplexity'], prog_bar=True, logger=True, on_step=True, on_epoch=True)
            
            return recon_loss, x_recon, vq_output, aeloss, perceptual_loss, gan_feat_loss

        if optimizer_idx == 1:
            # Train discriminator
            logits_image_real, _ = self.image_discriminator(frames.detach(), condition_emb)
            logits_video_real, _ = self.video_discriminator(x.detach(), condition_emb)

            logits_image_fake, _ = self.image_discriminator(frames_recon.detach(), condition_emb)
            logits_video_fake, _ = self.video_discriminator(x_recon.detach(), condition_emb)

            d_image_loss = self.disc_loss(logits_image_real, logits_image_fake)
            d_video_loss = self.disc_loss(logits_video_real, logits_video_fake)
            
            disc_factor = adopt_weight(self.global_step, threshold=self.cfg.model.discriminator_iter_start)
            discloss = disc_factor * (self.image_gan_weight*d_image_loss + self.video_gan_weight*d_video_loss)

            # Log metrics
            self.log("train/logits_image_real", logits_image_real.mean().detach(), logger=True, on_step=True, on_epoch=True)
            self.log("train/logits_image_fake", logits_image_fake.mean().detach(), logger=True, on_step=True, on_epoch=True)
            self.log("train/logits_video_real", logits_video_real.mean().detach(), logger=True, on_step=True, on_epoch=True)
            self.log("train/logits_video_fake", logits_video_fake.mean().detach(), logger=True, on_step=True, on_epoch=True)
            self.log("train/d_image_loss", d_image_loss, logger=True, on_step=True, on_epoch=True)
            self.log("train/d_video_loss", d_video_loss, logger=True, on_step=True, on_epoch=True)
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            
            return discloss

        perceptual_loss = self.perceptual_model(frames, frames_recon) * self.perceptual_weight
        return recon_loss, x_recon, vq_output, perceptual_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch['data']
        condition = batch['condition']  # Assuming condition is provided in the batch
        
        if optimizer_idx == 0:
            recon_loss, _, vq_output, aeloss, perceptual_loss, gan_feat_loss = self.forward(
                x, condition, optimizer_idx)
            commitment_loss = vq_output['commitment_loss']
            loss = recon_loss + commitment_loss + aeloss + perceptual_loss + gan_feat_loss
        if optimizer_idx == 1:
            discloss = self.forward(x, condition, optimizer_idx)
            loss = discloss
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['data']
        condition = batch['condition']
        
        recon_loss, _, vq_output, perceptual_loss = self.forward(x, condition)
        self.log('val/recon_loss', recon_loss, prog_bar=True)
        self.log('val/perceptual_loss', perceptual_loss, prog_bar=True)
        self.log('val/perplexity', vq_output['perplexity'], prog_bar=True)
        self.log('val/commitment_loss', vq_output['commitment_loss'], prog_bar=True)

    def configure_optimizers(self):
        lr = self.cfg.model.lr
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.pre_vq_conv.parameters()) +
                                  list(self.post_vq_conv.parameters()) +
                                  list(self.codebook.parameters()) +
                                  list(self.condition_embedding.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(list(self.image_discriminator.parameters()) +
                                    list(self.video_discriminator.parameters()),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def log_images(self, batch, **kwargs):
        log = dict()
        x = batch['data']
        condition = batch['condition']
        x = x.to(self.device)
        condition = condition.to(self.device)
        frames, frames_rec, _, _ = self(x, condition, log_image=True)
        log["inputs"] = frames
        log["reconstructions"] = frames_rec
        log["condition"] = condition
        return log

    def log_videos(self, batch, **kwargs):
        log = dict()
        x = batch['data']
        condition = batch['condition']
        x = x.to(self.device)
        condition = condition.to(self.device)
        _, _, x, x_rec = self(x, condition, log_image=True)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        log["condition"] = condition
        return log


def Normalize(in_channels, norm_type='group', num_groups=32):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return torch.nn.SyncBatchNorm(in_channels)


class ConditionalEncoder(nn.Module):
    def __init__(self, n_hiddens, downsample, image_channel=3, norm_type='group', padding_type='replicate', num_groups=32, condition_dim=64):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.conv_blocks = nn.ModuleList()
        max_ds = n_times_downsample.max()

        # First convolution without conditioning
        self.conv_first = SamePadConv3d(
            image_channel, n_hiddens, kernel_size=3, padding_type=padding_type)
        
        # Film conditioning layers for each block
        self.film_layers = nn.ModuleList()

        for i in range(max_ds):
            block = nn.Module()
            in_channels = n_hiddens * 2**i
            out_channels = n_hiddens * 2**(i+1)
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            
            block.down = SamePadConv3d(
                in_channels, out_channels, 4, stride=stride, padding_type=padding_type)
            
            block.res = ConditionalResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups, condition_dim=condition_dim)
                
            self.conv_blocks.append(block)
            n_times_downsample -= 1
            
            # Add FiLM layer for this block
            self.film_layers.append(FiLMLayer(out_channels, condition_dim))

        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type, num_groups=num_groups),
            SiLU()
        )

        self.out_channels = out_channels

    def forward(self, x, condition):
        """
        Forward pass with conditioning
        
        Args:
            x: Input tensor [B, C, T, H, W]
            condition: Condition embedding [B, condition_dim]
            
        Returns:
            Encoded representation with conditioning applied
        """
        h = self.conv_first(x)
        
        for i, (block, film) in enumerate(zip(self.conv_blocks, self.film_layers)):
            h = block.down(h)
            h = block.res(h, condition)
            h = film(h, condition)
            
        h = self.final_block(h)
        return h


class ConditionalDecoder(nn.Module):
    def __init__(self, n_hiddens, upsample, image_channel, norm_type='group', num_groups=32, condition_dim=64):
        super().__init__()

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()

        in_channels = n_hiddens*2**max_us
        self.final_block = nn.Sequential(
            Normalize(in_channels, norm_type, num_groups=num_groups),
            SiLU()
        )

        self.conv_blocks = nn.ModuleList()
        
        # Film conditioning layers for each block
        self.film_layers = nn.ModuleList()
        
        for i in range(max_us):
            block = nn.Module()
            in_channels = in_channels if i == 0 else n_hiddens*2**(max_us-i+1)
            out_channels = n_hiddens*2**(max_us-i)
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            
            block.up = SamePadConvTranspose3d(
                in_channels, out_channels, 4, stride=us)
                
            block.res1 = ConditionalResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups, condition_dim=condition_dim)
                
            block.res2 = ConditionalResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups, condition_dim=condition_dim)
                
            self.conv_blocks.append(block)
            n_times_upsample -= 1
            
            # Add FiLM layer for this block
            self.film_layers.append(FiLMLayer(out_channels, condition_dim))

        self.conv_last = SamePadConv3d(
            out_channels, image_channel, kernel_size=3)

    def forward(self, x, condition):
        """
        Forward pass with conditioning
        
        Args:
            x: Input tensor
            condition: Condition embedding [B, condition_dim]
            
        Returns:
            Decoded output with conditioning applied
        """
        h = self.final_block(x)
        
        for i, (block, film) in enumerate(zip(self.conv_blocks, self.film_layers)):
            h = block.up(h)
            h = block.res1(h, condition)
            h = block.res2(h, condition)
            h = film(h, condition)
            
        h = self.conv_last(h)
        return h


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer for conditioning
    """
    def __init__(self, num_features, condition_dim):
        super().__init__()
        self.num_features = num_features
        self.condition_proj = nn.Linear(condition_dim, num_features * 2)  # For scale and bias

    def forward(self, x, condition):
        params = self.condition_proj(condition)
        params = params.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [B, num_features*2, 1, 1, 1]
        
        scale, bias = torch.chunk(params, 2, dim=1)
        
        # Apply scale and bias
        return x * (scale + 1.0) + bias


class ConditionalResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, 
                 norm_type='group', padding_type='replicate', num_groups=32, condition_dim=64):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv1 = SamePadConv3d(
            in_channels, out_channels, kernel_size=3, padding_type=padding_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm2 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv2 = SamePadConv3d(
            out_channels, out_channels, kernel_size=3, padding_type=padding_type)
            
        if self.in_channels != self.out_channels:
            self.conv_shortcut = SamePadConv3d(
                in_channels, out_channels, kernel_size=3, padding_type=padding_type)
            
        # Add FiLM layers for conditioning
        self.film1 = FiLMLayer(in_channels, condition_dim)
        self.film2 = FiLMLayer(out_channels, condition_dim)

    def forward(self, x, condition):
        h = x
        h = self.norm1(h)
        h = self.film1(h, condition)
        h = silu(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = self.film2(h, condition)
        h = silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x + h


# Same padding convolution layers
class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # Assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input, mode=self.padding_type))


class SamePadConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.convt = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                        stride=stride, bias=bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input, mode=self.padding_type))


class ConditionalNLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, 
                 use_sigmoid=False, getIntermFeat=True, condition_dim=64):
        super(ConditionalNLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        self.condition_dim = condition_dim

        # Projection for condition embedding
        self.condition_proj = nn.Linear(condition_dim, ndf)

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc + 1, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]
        # +1 for the channel-wise condition

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input, condition):
        # Process condition
        batch_size = input.size(0)
        condition_mapped = self.condition_proj(condition)
        
        # Expand condition to spatial dimensions
        h, w = input.size(2), input.size(3)
        condition_spatial = condition_mapped.view(batch_size, 1, 1, 1).expand(batch_size, 1, h, w)
        
        # Concatenate input and condition
        input_with_condition = torch.cat([input, condition_spatial], dim=1)
        
        if self.getIntermFeat:
            res = [input_with_condition]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input_with_condition), None


class ConditionalNLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, 
                 use_sigmoid=False, getIntermFeat=True, condition_dim=64):
        super(ConditionalNLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        self.condition_dim = condition_dim

        # Projection for condition embedding
        self.condition_proj = nn.Linear(condition_dim, ndf)

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv3d(input_nc + 1, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]
        # +1 for the channel-wise condition

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input, condition):
        # Process condition
        batch_size = input.size(0)
        condition_mapped = self.condition_proj(condition)
        
        # Expand condition to spatial dimensions
        t, h, w = input.size(2), input.size(3), input.size(4)
        condition_spatial = condition_mapped.view(batch_size, 1, 1, 1, 1).expand(batch_size, 1, t, h, w)
        
        # Concatenate input and condition
        input_with_condition = torch.cat([input, condition_spatial], dim=1)
        
        if self.getIntermFeat:
            res = [input_with_condition]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input_with_condition), None