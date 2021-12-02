"""
Credit to: https://github.com/naoto0804/pytorch-AdaIN
"""
import torch
import torch.nn as nn

from .net import decoder, vgg
from .function import adaptive_instance_normalization


class UndoNorm:
    """Denormalize batch images."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Input:
            tensor (torch.Tensor): tensor image of size (B, C, H, W)
        """
        tensor = tensor.permute(1, 0, 2, 3)  # to (C, B, H, W)
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        tensor = tensor.permute(1, 0, 2, 3)  # to (B, C, H, W)
        return tensor


class Norm:
    """Normalize batch images."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Input:
            tensor (torch.Tensor): tensor image of size (B, C, H, W)
        """
        tensor = tensor.permute(1, 0, 2, 3)  # to (C, B, H, W)
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        tensor = tensor.permute(1, 0, 2, 3)  # to (B, C, H, W)
        return tensor


class AdaIN:
    """Adaptive Instance Normalization.

    https://github.com/KaiyangZhou/ssdg-benchmark

    https://arxiv.org/abs/2106.00592

    Reference:
        - Huang and Belongie. Arbitrary Style Transfer in Real-Time With
        Adaptive Instance Normalization. ICCV, 2017.
        - Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(
        self,
        decoder_weights,
        vgg_weights,
        device,
        alpha=0.5,
        norm_mean=None,
        norm_std=None,
    ):
        """
        Args:
            decoder_weights (str): path to decoder's weights
            vgg_weights (str): path to vgg's weights
            device (torch.device): cuda or cpu
            alpha (float, optional): interpolation parameter within (0, 1], which can be
                changed in __call__()
            norm_mean (list, optional): normalization mean, if it is not None, it means
                the input content/style tensors were normalized
            norm_std (list, optional): normalization std, if it is not None, it means
                the input content/style tensors were normalized
        """
        assert 0 <= alpha <= 1
        self.device = device
        self.alpha = alpha

        self.undo_norm = None
        self.norm = None

        if norm_mean is not None and norm_std is not None:
            self.undo_norm = UndoNorm(norm_mean, norm_std)
            self.norm = Norm(norm_mean, norm_std)

        self.build_models(decoder_weights, vgg_weights)

    def build_models(self, decoder_weights, vgg_weights):
        print("Building vgg and decoder for style transfer")

        self.decoder = decoder
        self.vgg = vgg

        self.decoder.eval()
        self.vgg.eval()

        print(f"Loading decoder weights from {decoder_weights}")
        self.decoder.load_state_dict(torch.load(decoder_weights))

        print(f"Loading vgg weights from {vgg_weights}")
        self.vgg.load_state_dict(torch.load(vgg_weights))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])

        self.vgg.to(self.device)
        self.decoder.to(self.device)

        for param in self.vgg.parameters():
            param.requires_grad = False

        for param in self.decoder.parameters():
            param.requires_grad = False

    def __call__(self, content, style, alpha=None):
        """
        Input:
            content (torch.Tensor): content minibatch of size (B, C, H, W)
            style (torch.Tensor): style minibatch of size (B, C, H, W)
            alpha (float, optional): interpolation parameter within (0, 1]
        """
        vgg = self.vgg
        decoder = self.decoder
        alpha = self.alpha if alpha is None else alpha

        if self.undo_norm is not None:
            # Map pixel values to [0, 1]
            content = self.undo_norm(content)
            style = self.undo_norm(style)

        content_f = vgg(content)
        style_f = vgg(style)
        feat = adaptive_instance_normalization(content_f, style_f)
        feat = feat * alpha + content_f * (1 - alpha)
        stylized = decoder(feat)

        if self.norm is not None:
            # Normalize pixel values
            stylized = self.norm(stylized)

        return stylized
