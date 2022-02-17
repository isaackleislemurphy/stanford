### general imports
import os
import pickle
import gc
import random
import itertools
import datetime as dt
import numpy as np
import pandas as pd

# from scipy.stats import norm
from tqdm import tqdm, trange

### viz imports
import matplotlib.pyplot as plt
import seaborn as sns

### GCloud imports
# from google.colab import drive
# drive.mount('/content/drive')

### Torch imports
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

### personal imports
from etl import *
from constants import *

### configs
torch.manual_seed(49)
pd.set_option("chained_assignment", None)


def penalize_gradient(gradient):
    """
    Calculates penalty of gradient -- in terms of (||G||_2^2 - 1)^2.
    Standard WGAN gradient penalty
    """
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)
    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)
    # Penalize the mean squared distance of the gradient norms from 1
    penalty = torch.mean(torch.pow(gradient_norm - 1.0, 2))
    return penalty


def get_gen_loss(critic_fake_play):
    """
    Computes generator loss, given a fake play
    """
    return -torch.mean(critic_fake_play)


def get_crit_loss(critic_fake_play, critic_real_play, grad_pen, lambda_=1.0):
    """
    Computes critic loss given real plays, fake plays, a penalty value, and a penalty weight.
    """
    return (
        -get_gen_loss(critic_real_play)
        + -get_gen_loss(critic_fake_play)
        + torch.mul(lambda_, grad_pen)
    )


def make_grad_hook():
    """
    Function to keep track of gradients for visualization purposes,
    which fills the grads list when using model.apply(grad_hook).
    """
    grads = []

    def grad_hook(m):
        if (
            isinstance(m, nn.Conv1d)
            or isinstance(m, nn.ConvTranspose1d)
            or isinstance(m, nn.Conv2d)
            or isinstance(m, nn.ConvTranspose2d)
            or isinstance(m, nn.Linear)
        ):
            grads.append(m.weight.grad)

    return grads, grad_hook


def get_noise(n_samples, z_dim, device="cpu"):
    """
    Makes noise vecs to feed into generator.

    Args:
      n_samples: int
        the number of samples to generate, a scalar
      z_dim: int
        the dimension of the noise vector, a scalar
      device: str
        the device type, usually 'cpu'

    Returns : torch.tensor(n_samples, z_dim)
      The random amples
    """
    return torch.randn(n_samples, z_dim, device=device)


class Generator(nn.Module):
    """
    Generator Class
    Attributes:
    """

    def __init__(
        self, z_dim=Z_DIM, play_len=N_FRAMES, play_width=FRAME_WIDTH, device="cpu"
    ):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.play_len = play_len
        self.play_width = play_width
        self.device = device

        # Build the neural network
        # linear mapping
        self.generator_lin = nn.Sequential(
            # self.make_gen_lin_block(z_dim, play_len * z_dim)
            self.make_gen_lin_block(z_dim, play_len),
            self.make_gen_lin_block(play_len, play_len * z_dim),
        )
        # convoluted mapping
        self.generator_conv = nn.Sequential(
            self.make_gen_conv_block(
                self.z_dim,
                play_len * 2,
                kernel_size=4,
                stride=1,
                final_layer=False,
                padding=2,
            ),
            self.make_gen_conv_block(
                play_len * 2,
                play_len * 4,
                kernel_size=4,
                stride=1,
                final_layer=False,
                padding=2,
            ),
            self.make_gen_conv_block(
                play_len * 4,
                play_len * 8,
                kernel_size=4,
                stride=1,
                final_layer=False,
                padding=2,
            ),
            self.make_gen_conv_block(
                play_len * 8,
                play_len * 4,
                kernel_size=4,
                stride=1,
                final_layer=False,
                padding=1,
            ),
            self.make_gen_conv_block(
                play_len * 4,
                play_len * 2,
                kernel_size=3,
                stride=1,
                final_layer=False,
                padding=1,
            ),
            self.make_gen_conv_block(
                play_len * 2, FRAME_WIDTH, kernel_size=3, stride=1, final_layer=True
            ),
        )

    def make_gen_lin_block(self, in_dim, out_dim, leaky_relu_activation=True):
        """
        Makes a linear layer for the initial linear pass, prior to convolution.

        Args:
          in_dim : int
            Input dimension
          out_dim : int
            Desired output dimension of linear layer
          leaky_relu_activation : bool
            Whether or not to add LeakyReLU activation
        """
        if leaky_relu_activation:
            return nn.Sequential(
                nn.Linear(in_dim, out_dim), nn.LeakyReLU(0.2, inplace=True)
            )
        return nn.Sequential(nn.Linear(in_dim, out_dim))

    def make_gen_conv_block(
        self,
        input_channels,
        output_channels,
        kernel_size=3,
        stride=2,
        final_layer=False,
        transpose=False,
        **kwargs,
    ):
        """
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        Parameters:
            input_channels : int
              how many channels the input feature representation has
            output_channels: int
              how many channels the output feature representation should have
            kernel_size: int or tuple
              the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: int
              the stride of the convolution
            final_layer: bool
              a boolean, true if it is the final layer and false otherwise
              (affects activation and batchnorm)
            transpose : bool
              If True, uses ConvTranspose1d; else Conv1D.
            **kwargs used to specify padding
        """
        conv = nn.ConvTranspose1d if transpose else nn.Conv1d
        if not final_layer:
            return nn.Sequential(
                conv(input_channels, output_channels, kernel_size, stride, **kwargs),
                nn.BatchNorm1d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                conv(input_channels, output_channels, kernel_size, stride, **kwargs),
                nn.Tanh(),
            )

    def forward(self, z_vec):
        """
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        """
        x = self.generator_lin(z_vec)
        x = x.view(z_vec.shape[0], self.z_dim, self.play_len)
        return self.generator_conv(x)


class Critic(nn.Module):
    """
    Critic Class
    Values:
        hidden_dim: the inner dimension, a scalar
    """

    def __init__(self, play_width=FRAME_WIDTH, play_len=N_FRAMES, hidden_dim=64):
        super(Critic, self).__init__()
        self.play_width = play_width
        self.play_len = play_len
        self.hidden_dim = hidden_dim
        self.critic = nn.Sequential(
            self.make_crit_block(
                self.play_width, self.hidden_dim, kernel_size=8, stride=2
            ),
            self.make_crit_block(
                self.hidden_dim, self.hidden_dim * 2, kernel_size=6, stride=2
            ),
            self.make_crit_block(
                self.hidden_dim * 2, self.hidden_dim * 2, kernel_size=6, stride=2
            ),
            self.make_crit_block(
                self.hidden_dim * 2, 1, kernel_size=4, stride=2, final_layer=True
            ),
        )

    def make_crit_block(
        self,
        input_channels,
        output_channels,
        kernel_size=4,
        stride=2,
        final_layer=False,
        **kwargs,
    ):
        """
        Function to return a sequence of operations corresponding to a critic block of DCGAN;
        a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm)
        """
        if not final_layer:
            return nn.Sequential(
                nn.Conv1d(
                    input_channels, output_channels, kernel_size, stride, **kwargs
                ),
                nn.BatchNorm1d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv1d(
                    input_channels, output_channels, kernel_size, stride, **kwargs
                ),
            )

    def forward(self, image):
        """
        Function for completing a forward pass of the critic: Given an image tensor,
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        """
        critic_pred = self.critic(image)
        return critic_pred.view(len(critic_pred), -1)


def compute_gradient(critic, real, fake, epsilon):
    """
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    """
    # Mix the images together
    mixed_plays = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = critic(mixed_plays)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=mixed_plays,
        outputs=mixed_scores,
        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def weights_init(model):
    """
    Initializes weights for a particular model.
    """
    if isinstance(model, nn.Conv1d) or isinstance(model, nn.ConvTranspose1d):
        torch.nn.init.normal_(model.weight, 0.0, 0.02)
    if isinstance(model, nn.BatchNorm1d):
        torch.nn.init.normal_(model.weight, 0.0, 0.02)
        torch.nn.init.constant_(model.bias, 0)


def plot_live_losses(generator_losses, critic_losses):
    """
    Plots generator and critic losses, as a check on convergence
    """
    plt.rcParams["figure.figsize"] = (16, 4)
    plt.plot(generator_losses, label="Generator Loss")
    plt.plot(critic_losses, label="Critic Loss")
    plt.legend()
    plt.title("Running Loss Convergence")
    plt.xlabel("Step")
    plt.ylabel("Loss Value")
    plt.show()


def recover_gradient_snapshot(gen, crit):
    """
    Gets min norm, mean norm, and max norm of
    generator and critic gradients, to check or vanishment.
    """
    # recover gradient statistics
    gen_grads, crit_grads = [], []
    for param in gen.parameters():
        gen_grads.append(param.grad.detach().norm().numpy())
    for param in crit.parameters():
        crit_grads.append(param.grad.detach().norm().numpy())
    gen_grads = np.stack(gen_grads).flatten()
    crit_grads = np.stack(crit_grads).flatten()

    return (
        [np.min(gen_grads), np.mean(gen_grads), np.max(gen_grads)],
        [np.min(crit_grads), np.mean(crit_grads), np.max(crit_grads)],
    )


def plot_live_gradients(grad_snapshots):
    """
    Plots running gradients, so you can manually inspect for vanishment.
    """
    result = np.stack(grad_snapshots)
    plt.rcParams["figure.figsize"] = (16, 5)
    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(
        result[:, 0, 0], label="Generator (Min)", color="orange", linestyle="dashed"
    )
    ax[0].plot(result[:, 0, 1], label="Generator (Mean)", color="orange")
    ax[0].plot(
        result[:, 0, 2], label="Generator (Max)", color="orange", linestyle="dotted"
    )
    ax[1].plot(result[:, 1, 0], label="Critic (Min)", color="blue", linestyle="dashed")
    ax[1].plot(result[:, 1, 1], label="Critic (Mean)", color="blue")
    ax[1].plot(result[:, 1, 2], label="Critic (Max)", color="blue", linestyle="dotted")

    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("|Grad|")
    plt.title("Running Gradient Behavior")
    plt.show()


def plot_demo_noise(z_vec_demo, gen):
    """
    Given a vector of noise, plots the generator's output
    at a current state. Helpful in tracking the convergence
    of the generator
    """
    plt.rcParams["figure.figsize"] = (16, 4)
    fig, ax = plt.subplots(ncols=z_vec_demo.shape[0])

    for k in range(z_vec_demo.shape[0]):
        route_tensor = (gen(z_vec_demo).detach().numpy())[k].T
        eval_frames = np.array(range(0, 40, 5))
        for j, i in enumerate(range(0, 24, 4)):
            ax[k].plot(
                route_tensor[eval_frames, i],
                route_tensor[eval_frames, i + 1],
                label=f"Route-Runner: {j}",
            )
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Example Plays")
    plt.show()
