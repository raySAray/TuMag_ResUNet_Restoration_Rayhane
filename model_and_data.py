import sys
sys.path.append(r"F:\desk2\projects IAA-CSIC\ML-noiseCansel-unet\02_psf_physics")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import psf_optics

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class ResUNetDeconv(nn.Module):
    """
    Residual U-Net for Solar Spectropolarimetric Deconvolution & Denoising.
    Predicts the residual correction over the Wiener-deconvolved input:
    out = x_wiener + UNet(x_wiener)
    """
    def __init__(self, in_channels=1, base_ch=32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_ch)
        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.bottleneck = DoubleConv(base_ch * 4, base_ch * 8)

        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, in_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        residual = self.out_conv(d1)
        return x + residual

class CharbonnierGradientLoss(nn.Module):
    """Combines smooth L1 (Charbonnier) loss with spatial gradient preservation."""
    def __init__(self, grad_weight=0.2, eps=1e-3):
        super().__init__()
        self.grad_weight = grad_weight
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        charb_loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))

        # Spatial gradients (dx, dy)
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        targ_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        targ_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        grad_loss = F.l1_loss(pred_dx, targ_dx) + F.l1_loss(pred_dy, targ_dy)
        return charb_loss + self.grad_weight * grad_loss

def apply_fft_blur_and_wiener(patch, psf, noise_sigma, wiener_K):
    """
    Forward degradation (PSF convolution + realistic photon noise)
    followed by Wiener deconvolution in Fourier space.
    """
    H, W = patch.shape
    psf_pad = np.zeros((H, W), dtype=np.float32)
    kh, kw = psf.shape
    psf_pad[:kh, :kw] = psf
    psf_pad = np.roll(psf_pad, (-kh // 2, -kw // 2), axis=(0, 1))

    OTF = np.fft.fft2(psf_pad)
    PATCH_F = np.fft.fft2(patch)

    # Convolve with PSF
    blurred = np.real(np.fft.ifft2(PATCH_F * OTF)).astype(np.float32)

    # Add realistic instrument noise
    if noise_sigma > 0:
        blurred += np.random.normal(0.0, noise_sigma, size=blurred.shape).astype(np.float32)

    # Wiener deconvolution
    BLUR_F = np.fft.fft2(blurred)
    W_filter = np.conj(OTF) / (np.abs(OTF)**2 + wiener_K)
    wiener_rec = np.real(np.fft.ifft2(BLUR_F * W_filter)).astype(np.float32)
    return wiener_rec

class TuMagStokesDataset(Dataset):
    """
    Extracts random patches from TuMag FITS data across wavelengths and Stokes parameters,
    normalizes each channel by its physical scale, and applies Zernike PSF + Wiener filtering.
    """
    def __init__(self, data_4d, zernike_list, num_samples=2000, patch_size=128, is_train=True):
        self.data = data_4d  # shape: (8, 4, H_region, W_region)
        self.zernike_list = zernike_list
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.is_train = is_train

        # Precompute bank of physical Zernike PSFs for fast training
        self.psf_bank = [
            psf_optics.generate_tumag_psf(psf_size=65, zernike_coeffs=z)
            for z in zernike_list
        ]

        # Measured physical noise sigmas and SNR-matched Wiener K per Stokes (I, Q, U, V)
        self.noise_sigmas = [0.0018, 0.0006, 0.00065, 0.00065]
        self.wiener_Ks = [1e-3, 5e-2, 5e-2, 5e-3]

        # Precompute channel-wise normalization stats (mean, std) per (wave, stokes)
        self.means = np.mean(self.data, axis=(2, 3), keepdims=False)
        self.stds = np.std(self.data, axis=(2, 3), keepdims=False) + 1e-6

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        nw, ns, H, W = self.data.shape
        w_idx = np.random.randint(0, nw)
        # Weight Stokes I and V slightly higher, while still training on Q and U
        s_idx = np.random.choice([0, 1, 2, 3], p=[0.35, 0.15, 0.15, 0.35])

        # Extract patch with 16px margin to avoid circular FFT boundary wrap-around
        margin = 16
        full_size = self.patch_size + 2 * margin
        y = np.random.randint(0, H - full_size)
        x = np.random.randint(0, W - full_size)

        gt_full = self.data[w_idx, s_idx, y:y+full_size, x:x+full_size].copy()

        # Random geometric augmentation (flips & 90-deg rotations)
        if self.is_train:
            k_rot = np.random.randint(0, 4)
            gt_full = np.rot90(gt_full, k_rot).copy()
            if np.random.rand() > 0.5:
                gt_full = np.fliplr(gt_full).copy()

        # Pick a random realistic Zernike PSF from valid QSUN observations
        psf = self.psf_bank[np.random.randint(0, len(self.psf_bank))]
        wiener_full = apply_fft_blur_and_wiener(
            gt_full,
            psf=psf,
            noise_sigma=self.noise_sigmas[s_idx],
            wiener_K=self.wiener_Ks[s_idx]
        )

        # Crop central patch_size x patch_size (border-artifact free)
        gt_crop = gt_full[margin:-margin, margin:-margin]
        wiener_crop = wiener_full[margin:-margin, margin:-margin]

        # Standardize using channel-specific physical stats
        m = self.means[w_idx, s_idx]
        s = self.stds[w_idx, s_idx]
        gt_norm = (gt_crop - m) / s
        wiener_norm = (wiener_crop - m) / s

        return (
            torch.from_numpy(wiener_norm).unsqueeze(0),
            torch.from_numpy(gt_norm).unsqueeze(0)
        )
