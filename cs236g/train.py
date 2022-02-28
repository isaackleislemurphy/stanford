"""Trains the model"""
### Torch imports
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler

# do you want to re-read data
READ_DATA = False

from constants import *
from etl import *
from model import *


### example configs
CONFIGS = Configs(
    n_epochs=100,
    z_dim=Z_DIM,
    display_step=1,
    save_step=1,
    batch_size=512,
    lr=2e-4,  # 0.0002
    beta_1=0.5,
    beta_2=0.999,
    c_lambda=10,
    crit_repeats=5,
    device="cpu",
    z_vec_demo=torch.hstack(
        [
            get_noise(25, Z_DIM),
            torch.from_numpy(
                z_supp_train[
                    np.random.choice(range(z_supp_train.shape[0]), 25), :
                ].astype("float32")
            ),
        ]
    ),
)


def main(configs=CONFIGS):
    """
    Trains the model.

    Args:
      configs : namedtuple
        A named tuple of configs; see the default example above. Results saved to drive.

    TODO: break this up into something more modular, e.g. ingest + train or something like that.
    """

    ########################################################################################
    # Ingest
    ########################################################################################
    if READ_DATA:
        # read train + val
        routes_train, play_info_train, failures_train = fetch_data(
            week_start=2, week_end=13
        )
        z_supp_train = np.vstack(
            [make_play_z_inits(item) for item in tqdm(play_info_train)]
        )

        routes_dev, play_info_dev, failures_dev = fetch_data(week_start=14, week_end=15)

        save_pickle(routes_train, FILEPATH + "routes_train.pkl")
        save_pickle(play_info_train, FILEPATH + "play_info_train.pkl")
        save_pickle(failures_train, FILEPATH + "failures_train.pkl")
        save_pickle(z_supp_train, FILEPATH + "z_supp_train.pkl")

        save_pickle(routes_dev, FILEPATH + "routes_dev.pkl")
        save_pickle(play_info_dev, FILEPATH + "play_info_dev.pkl")
        save_pickle(failures_dev, FILEPATH + "failures_dev.pkl")

    else:
        routes_train = load_pickle(FILEPATH + "routes_train.pkl")
        routes_dev = load_pickle(FILEPATH + "routes_dev.pkl")
        z_supp_train = load_pickle(FILEPATH + "z_supp_train.pkl")

        play_info_train = load_pickle(FILEPATH + "play_info_train.pkl")
        play_info_dev = load_pickle(FILEPATH + "play_info_dev.pkl")

    z_scaler = StandardScaler()
    # Z-scale the starting positions
    z_supp_train[:, 0:24] = z_scaler.fit_transform(z_supp_train[:, 0:24])

    ########################################################################################
    # Configure storage
    ########################################################################################
    # name the run
    run_id = f"lr{configs.lr}_clambda{configs.c_lambda}_cr{configs.crit_repeats}_bs{configs.batch_size}_z{configs.z_dim}"
    ### make the folder
    make_run_folders(run_id=run_id, configs=configs, runpath=RUNPATH)

    ########################################################################################
    # Setup GPU/CPU
    ########################################################################################
    ### check for device
    if torch.cuda.is_available():
        print("%" * 40)
        print("Cuda available")
        device = "cuda"
        torch.cuda.empty_cache()
    else:
        device = configs.device

    ########################################################################################
    # Build + Initialize models
    ########################################################################################
    ### construct models ###
    ### models
    gen = Generator(configs.z_dim + Z_SUPP_DIM).to(device)
    crit = Critic().to(device)
    ### optimizers
    gen_opt = torch.optim.Adam(
        gen.parameters(), lr=configs.lr, betas=(configs.beta_1, configs.beta_2)
    )
    crit_opt = torch.optim.Adam(
        crit.parameters(), lr=configs.lr, betas=(configs.beta_1, configs.beta_2)
    )
    ### weights
    gen = gen.apply(weights_init)
    crit = crit.apply(weights_init)

    ########################################################################################
    # Initialize data
    ########################################################################################
    ### load data
    data_loader = torch.utils.data.DataLoader(
        dataset=TensorDataset(
            torch.from_numpy(routes_train.astype("float32")).to(device),
            torch.from_numpy(z_supp_train.astype("float32")).to(device),
        ),
        batch_size=configs.batch_size,
        shuffle=True,
    )
    ########################################################################################
    # Fit the model
    ########################################################################################
    ### fit model
    cur_step = 0
    generator_losses, critic_losses, grad_snapshots = [], [], []
    for epoch in range(configs.n_epochs):
        print(f"Epoch: {epoch + 1}/{configs.n_epochs}")
        # Dataloader returns the batches
        for (real, z_start) in tqdm(data_loader):
            cur_batch_size = len(real)
            real = real.to(device)
            mean_iteration_critic_loss = 0
            for k in range(configs.crit_repeats):
                ### Update critic ###
                crit_opt.zero_grad()
                # fake_noise = get_noise(cur_batch_size, z_dim, device=device)
                fake_noise = torch.hstack(
                    [get_noise(cur_batch_size, configs.z_dim, device=device), z_start]
                )
                fake = gen(fake_noise)
                crit_fake_pred = crit(fake.detach())
                crit_real_pred = crit(real)

                epsilon = torch.rand(len(real), 1, 1, device=device, requires_grad=True)
                gradient = compute_gradient(crit, real, fake.detach(), epsilon)
                gp = penalize_gradient(gradient)
                crit_loss = get_crit_loss(
                    crit_fake_pred, crit_real_pred, gp, configs.c_lambda
                )

                # Keep track of the average critic loss in this batch
                mean_iteration_critic_loss += crit_loss.item() / configs.crit_repeats
                # Update gradients
                crit_loss.backward(retain_graph=True)
                # Update optimizer
                crit_opt.step()
            critic_losses += [mean_iteration_critic_loss]

            ### Update generator ###
            gen_opt.zero_grad()
            fake_noise_2 = torch.hstack(
                [get_noise(cur_batch_size, configs.z_dim, device=device), z_start]
            )
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

        if ((epoch + 1) % configs.save_step) == 0:
            gen.to("cpu")
            # save state
            pd.DataFrame({"num_epochs_completed": [epoch + 1]}).to_csv(
                RUNPATH + run_id + "/cur_epochs.csv"
            )
            # save models
            save_pickle(gen, RUNPATH + run_id + "/weights/gen.pkl")
            save_pickle(crit, RUNPATH + run_id + "/weights/crit.pkl")
            save_pickle(gen_opt, RUNPATH + run_id + "/weights/gen_opt.pkl")
            save_pickle(crit_opt, RUNPATH + run_id + "/weights/crit_opt.pkl")
            torch.save(gen.state_dict(), RUNPATH + run_id + "/weights/gen_weights.pth")
            torch.save(
                crit.state_dict(), RUNPATH + run_id + "/weights/crit_weights.pth"
            )
            # save losses
            save_pickle(
                generator_losses,
                RUNPATH + run_id + "/history/generator_losses_train.pkl",
            )
            save_pickle(
                critic_losses, RUNPATH + run_id + "/history/critic_losses_train.pkl"
            )
            plot_live_losses(
                generator_losses,
                critic_losses,
                filename=RUNPATH + run_id + "/plots/train_losses.png",
            )
            plot_live_gradients(
                grad_snapshots, filename=RUNPATH + run_id + "/plots/grads.png"
            )
            plot_demo_noise(
                configs.z_vec_demo.cpu(),
                gen,
                filename=RUNPATH
                + run_id
                + f"/plots/sample_images/sample_images_{epoch}.png",
            )
            gen.to(device)


if __name__ == "__main__":
    main()
