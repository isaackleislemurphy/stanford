"""Functions useful for modeling"""

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


def invert_scale(img, scale=1, offset=0):
    """
    Returns a variable to its original scale, for the purpose
    of enforcing kinematic equations.

    Args:
      img : torch.tensor
        A tensor or slice of tensor to be inverse scaled
      scale : float
        scaling value used in scaling
      offset : float
        centering value used in scaling
    Returns : torch.tensor
      The tensor img, but inverse-scaled
    """
    return torch.add(
        torch.multiply(torch.multiply(torch.add(img, 1), 0.5), scale), offset
    )


def penalize_initial_positions(real, fake, dim=[0, 1]):
    """ """
    x_init_pen = torch.pow(
        torch.add(
            fake[:, list(np.arange(0, 24, 4)), 0],
            -real[:, list(np.arange(0, 24, 4)), 0],
        ),
        2,
    )
    y_init_pen = torch.pow(
        torch.add(
            fake[:, list(np.arange(1, 24, 4)), 0],
            -real[:, list(np.arange(1, 24, 4)), 0],
        ),
        2,
    )
    return torch.add(torch.mean(x_init_pen, dim=dim), torch.mean(y_init_pen, dim=dim))


def eval_kinematic(img, buffer_pos=0.1, buffer_vel=0.1):
    """ """
    img_inv_sc = torch.clone(img)
    pos_slice = list(np.sort(np.hstack([np.arange(0, 24, 4), np.arange(1, 24, 4)])))

    # undo x/y scaling
 #   x = invert_scale(img_inv_sc[:, list(np.arange(0, 24, 4)), :], 90, -45)
 #   y = invert_scale(img_inv_sc[:, list(np.arange(1, 24, 4)), :], 70, -15)
    x = img_inv_sc[:, list(np.arange(0, 24, 4)), :] * 11.6
    y = img_inv_sc[:, list(np.arange(1, 24, 4)), :] * 25
    v = invert_scale(img_inv_sc[:, list(np.arange(2, 24, 4)), :], 12, 0)
    a = invert_scale(img_inv_sc[:, list(np.arange(2, 24, 4)), :], 12, 0)

    # calculate displacement
    squared_xy_delta = torch.pow(
        torch.add(
            torch.pow(torch.add(x[:, :, 1:], -x[:, :, :-1]), 2),
            torch.pow(torch.add(y[:, :, 1:], -y[:, :, :-1]), 2),
        ),
        1 / 2,
    )

    # calculate v - v0; timestep length cancels
    v_delta = torch.multiply(torch.add(v[:, :, 1:], v[:, :, :-1]), 0.1 / 2)
    # violations of \delta_x = t(v + v_0)/2 equation
    kinematic_violations_position = squared_xy_delta > v_delta + buffer_pos
    # violations of \delta_v = v_0 + a_0 t
    kinematic_violations_velo = v[:, :, 1:] > torch.add(
        torch.add(v[:, :, :-1], torch.multiply(a[:, :, :-1], 0.1)), buffer_vel
    )

    return (kinematic_violations_position, kinematic_violations_velo)


def penalize_kinematic(img, buffer_pos=0.1, buffer_vel=0.1, dim=[0, 1, 2]):
    """
    Penalizes an image for each pixel it violates the kinematic equations.

    """
    kin_violations_pos, kin_violations_velo = eval_kinematic(
        img, buffer_pos, buffer_vel
    )

    return torch.add(
        torch.mean(kin_violations_pos.float(), dim=dim),
        torch.mean(kin_violations_velo.float(), dim=dim),
    )


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


def get_gen_loss(
    critic_fake_play,
    init_pos_pen=0.0,
    kinematic_pen=0.0,
    lambda_init_pos=0.0,
    lambda_kinematic=0.0,
):
    """
    Computes generator loss, given a fake play
    """
    return (
        -torch.mean(critic_fake_play)
        + torch.mul(lambda_init_pos, init_pos_pen)
        + torch.mul(lambda_kinematic, kinematic_pen)
    )


def get_crit_loss(
    critic_fake_play,
    critic_real_play,
    grad_pen,
    lambda_grad=10.0,
):
    """
    Computes critic loss given real plays, fake plays, a penalty value, and a penalty weight.
    TODO: Update docstring here
    """
    return (
        -get_gen_loss(critic_real_play)
        - get_gen_loss(critic_fake_play)
        + torch.mul(lambda_grad, grad_pen)
        # + torch.mul(lambda_init_pos, init_pos_pen)
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
        self,
        z_dim=Z_DIM + Z_SUPP_DIM,
        play_len=N_FRAMES,
        play_width=FRAME_WIDTH,
        device="cpu",
    ):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.play_len = play_len
        self.play_width = play_width
        self.device = device

        # Build the neural network
        # linear mapping
        self.generator_lin = nn.Sequential(
            self.make_gen_lin_block(z_dim, 2 * play_len * z_dim),
            self.make_gen_lin_block(2 * z_dim * play_len, play_len * z_dim),
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
                nn.LeakyReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                conv(input_channels, output_channels, kernel_size, stride, **kwargs),
#                nn.Tanh(),
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

    def __init__(
        self, play_width=FRAME_WIDTH, play_len=N_FRAMES, hidden_dim=64, img_supp_dim=2
    ):
        super(Critic, self).__init__()
        self.play_width = play_width
        self.play_len = play_len
        self.hidden_dim = hidden_dim
        self.img_supp_dim = img_supp_dim
        self.critic_conv = nn.Sequential(
            self.make_crit_conv_block(
                self.play_width, self.hidden_dim, kernel_size=4, stride=2, padding=2
            ),
            self.make_crit_conv_block(
                self.hidden_dim, self.hidden_dim * 2, kernel_size=4, stride=2, padding=2
            ),
            self.make_crit_conv_block(
                self.hidden_dim * 2,
                self.hidden_dim * 4,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
            self.make_crit_conv_block(self.hidden_dim * 4, 1, kernel_size=4, stride=1),
        )
        self.critic_lin = nn.Sequential(
            # self.make_crit_lin_block(self.play_len + self.img_supp_dim, 256, True),
            self.make_crit_lin_block(16 + self.img_supp_dim, 256, True),
            self.make_crit_lin_block(256, 128, True),
            self.make_crit_lin_block(128, 1, False),
        )

    def make_crit_lin_block(self, in_dim, out_dim, leaky_relu_activation=True):
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

    def make_crit_conv_block(
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

    def forward(self, image, img_supp):
        """
        Function for completing a forward pass of the critic: Given an image tensor,
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        """

        # put image through conv layers
        result_a = self.critic_conv(image).view(len(image), -1)

        # append img_supp
        result_b = torch.cat([result_a, img_supp], dim=1)

        # put through linear layers
        result_c = self.critic_lin(result_b)
        return result_c
        # return critic_pred.view(len(critic_pred), -1)


def compute_gradient(critic, real, fake, epsilon, *args, **kwargs):
    """
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
        **kwargs passed to critic
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    """
    # Mix the images together
    mixed_plays = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = critic(mixed_plays, *args, **kwargs)

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


def plot_live_losses(generator_losses, critic_losses, filename=None):
    """
    Plots generator and critic losses, as a check on convergence
    """
    plt.rcParams["figure.figsize"] = (16, 8)
    plt.plot(generator_losses, label="Generator Loss")
    plt.plot(critic_losses, label="Critic Loss")
    plt.legend()
    plt.title("Running Loss Convergence")
    plt.xlabel("Step")
    plt.ylabel("Loss Value")
    if filename is not None:
        assert isinstance(filename, str)
        plt.savefig(filename)
        plt.clf()
    else:
        plt.show()


def recover_gradient_snapshot(gen, crit):
    """
    Gets min norm, mean norm, and max norm of
    generator and critic gradients, to check or vanishment.
    """
    # recover gradient statistics
    gen_grads, crit_grads = [], []
    for param in gen.parameters():
        gen_grads.append(param.grad.detach().cpu().norm().numpy())
    for param in crit.parameters():
        crit_grads.append(param.grad.detach().cpu().norm().numpy())
    gen_grads = np.stack(gen_grads).flatten()
    crit_grads = np.stack(crit_grads).flatten()

    return (
        [np.min(gen_grads), np.mean(gen_grads), np.max(gen_grads)],
        [np.min(crit_grads), np.mean(crit_grads), np.max(crit_grads)],
    )


def plot_live_gradients(grad_snapshots, filename=None, full_plot=True):
    """
    Plots running gradients, so you can manually inspect for vanishment.
    """
    result = np.stack(grad_snapshots)
    plt.rcParams["figure.figsize"] = (16, 8)
    fig, ax = plt.subplots(nrows=2)
    
    if full_plot:
        ax[0].plot(
        result[:, 0, 0], label="Generator (Min)", color="orange", linestyle="dashed"
    )
    ax[0].plot(result[:, 0, 1], label="Generator (Mean)", color="orange")
    if full_plot:
        ax[0].plot(
        result[:, 0, 2], label="Generator (Max)", color="orange", linestyle="dotted"
    )
    if full_plot:
        ax[1].plot(result[:, 1, 0], label="Critic (Min)", color="blue", linestyle="dashed")
    ax[1].plot(result[:, 1, 1], label="Critic (Mean)", color="blue")
    if full_plot:
        ax[1].plot(result[:, 1, 2], label="Critic (Max)", color="blue", linestyle="dotted")

    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("|Grad|")
    plt.title("Running Gradient Behavior")
    if filename is not None:
        assert isinstance(filename, str)
        plt.savefig(filename)
        plt.clf()
    else:
        plt.show()


def plot_demo_noise(z_vec_demo, gen, filename=None):
    """
    Given a vector of noise, plots the generator's output
    at a current state. Helpful in tracking the convergence
    of the generator
    """
    route_runner_colors = ["blue", "orange", "red", "green", "pink", "brown"]
    plt.rcParams["figure.figsize"] = (12, 12) if filename is None else (25, 25)
    fig, ax = plt.subplots(ncols=5, nrows=5)

    ctr_row, ctr_col = 0, 0
    for k in range(z_vec_demo.shape[0]):
        route_tensor = (gen(z_vec_demo).detach().numpy())[k].T
        eval_frames = np.array(range(0, 75, 1))
        for j, i in enumerate(range(0, 24, 4)):
            # plot route
            ax[ctr_row, ctr_col].plot(
                route_tensor[eval_frames, i],
                route_tensor[eval_frames, i + 1],
                label=f"Route-Runner: {j}" if j < 5 else "QB",
                c=route_runner_colors[j],
                alpha=0.5,
            )
            ax[ctr_row, ctr_col].scatter(
                route_tensor[0, i], route_tensor[0, i + 1], c="black", marker="^"
            )
        if ctr_col == 4:
            ctr_row += 1
            ctr_col = 0
        else:
            ctr_col += 1
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Example Plays")
    if filename is not None:
        assert isinstance(filename, str)
        plt.savefig(filename)
        plt.clf()
    else:
        plt.show()


def make_run_folders(run_id, configs, runpath=RUNPATH):
    """
    Makes a folder in drive in which to store your runs
    """
    os.mkdir(RUNPATH + run_id)
    # directory to run folder
    run_id_path = RUNPATH + run_id + "/"
    # make relevant folders
    os.mkdir(run_id_path + "plots/")
    os.mkdir(run_id_path + "plots/sample_images")
    os.mkdir(run_id_path + "weights/")
    os.mkdir(run_id_path + "history/")
    save_pickle(configs, run_id_path + "configs.pkl")
