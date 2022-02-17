"""
Trains the baseline model.
"""

### Torch imports
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

from constants import *
from etl import *
from model import *


def main():
    ### ingest and process data ###
    routes, play_info, failures = fetch_data(week_start=2, week_end=2)

    ### initialize hyperparams ###
    # TODO: move this all to a configs
    n_epochs = 100
    z_dim = Z_DIM
    display_step = 5
    batch_size = 128
    lr = 1e-4  # 0.0002
    beta_1 = 0.5
    beta_2 = 0.999
    c_lambda = 10
    crit_repeats = 10
    device = "cpu"
    z_vec_demo = get_noise(4, z_dim)

    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.empty_cache()

    ### construct models ###
    ### generator
    gen = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    ### critic
    crit = Critic().to(device)
    crit_opt = torch.optim.Adam(crit.parameters(), lr=lr, betas=(beta_1, beta_2))
    ### construct weights
    gen = gen.apply(weights_init)
    crit = crit.apply(weights_init)

    ### load data
    data_loader = torch.utils.data.DataLoader(
        dataset=torch.from_numpy(routes.astype("float32")),
        batch_size=batch_size,
        shuffle=True,
    )
    ### fit model ###
    cur_step = 0
    generator_losses, critic_losses, grad_snapshots = [], [], []
    for epoch in range(n_epochs):
        print(f"Epoch: {epoch + 1}/{n_epochs}")
        # Dataloader returns the batches
        for real in tqdm(data_loader):
            cur_batch_size = len(real)
            real = real.to(device)
            mean_iteration_critic_loss = 0
            for k in range(crit_repeats):
                ### Update critic ###
                crit_opt.zero_grad()
                fake_noise = get_noise(cur_batch_size, z_dim, device=device)
                fake = gen(fake_noise)
                crit_fake_pred = crit(fake.detach())
                crit_real_pred = crit(real)

                epsilon = torch.rand(len(real), 1, 1, device=device, requires_grad=True)
                gradient = compute_gradient(crit, real, fake.detach(), epsilon)
                gp = penalize_gradient(gradient)
                crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)

                # Keep track of the average critic loss in this batch
                mean_iteration_critic_loss += crit_loss.item() / crit_repeats
                # Update gradients
                crit_loss.backward(retain_graph=True)
                # Update optimizer
                crit_opt.step()
            critic_losses += [mean_iteration_critic_loss]

            ### Update generator ###
            gen_opt.zero_grad()
            fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
            fake_2 = gen(fake_noise_2)
            crit_fake_pred = crit(fake_2)

            gen_loss = get_gen_loss(crit_fake_pred)
            gen_loss.backward(retain_graph=True)

            # Update the weights
            gen_opt.step()

            # Keep track of the average generator loss
            generator_losses += [gen_loss.item()]

            # Keep track of gradient behavior
            grad_snapshot = recover_gradient_snapshot(gen, crit)
            grad_snapshots.append(grad_snapshot)
            cur_step += 1
        if ((epoch + 1) % display_step) == 0:
            plot_live_losses(generator_losses, critic_losses)
            plot_live_gradients(grad_snapshots)
            plot_demo_noise(z_vec_demo, gen)


if __name__ == "__main__":
    main()
