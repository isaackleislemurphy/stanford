from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import get_logger
from utils.test_env import EnvTest
from q4_schedule import LinearExploration, LinearSchedule
from q5_linear_torch import Linear


from configs.q6_nature import config


def do_conv2d_dim_math(dim_in, layer, axis=0):
    """
    Given a Conv2D layer and all its bells and whistles,
    this function figures out what the resulting "image" will look like.

    Args:
        dim_in : int
            Dimension of image along axis specified by `axis`.
        layer : torch.nn.Conv2D
            A Conv2D layer that's convolving stuff
        axis : int
            Axis along which to compute the dim change

    Returns : int
        Dimension of new image along specified axis.
    """
    # make sure everything's accounted for
    for attribute in ("kernel_size", "stride", "padding", "dilation"):
        assert attribute in dir(layer)
    # unpack conv2 construction
    kernel_size, stride, padding, dilation = (
        layer.kernel_size[axis],
        layer.stride[axis],
        layer.padding[axis],
        layer.dilation[axis]
    )
    # calculate new dims
    dim_out = (dim_in - kernel_size + 2 * padding) / stride + 1
    print(f"   Dim In/Partial/Out: {dim_in}, {dim_out}, {int(dim_out)}")
    return int(dim_out)

# to solve the padding, as instructed by problem
solve_pad = lambda stride, img_height, filter_size: ((stride - 1) * img_height - stride + filter_size) // 2

def make_dqn_skeleton(state_shape, num_actions, config):
    """
    Makes an ordered dict for your DQN.
    These ordered dicts can then be passed to Torch's Sequential method()

    Docs: https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
    * See "...Alternatively, an OrderedDict of modules can be passed in...""
    That's what we're shooting for here

    Args:
        state_shape : tuple[int, int, int]
            Dimensions of the state shape
        num_actions : int
            Number of actions you could take
        config : dict
            Gym configs

    Returns : OrderDict
        A dictionary of torch layers, fully-prepared to be passed to Sequential
    """
    model_skeleton = OrderedDict()

    # have on hand
    img_height, img_width, n_channels = state_shape
    # for tracking the sizes of the images throughout
    active_height, active_width = img_height, img_width
    print("----- Model Skeleton ------")
    print((active_height, active_width))

    ######################################################
    # first convolution
    model_skeleton["conv_1"] = nn.Conv2d(
        in_channels=n_channels * config.state_history, # use the lag-4 channel structure
        out_channels=32, # first layer, per paper
        kernel_size=8,# first layer, per paper
        stride=4, # first layer, per paper
        padding=solve_pad(4, img_height, 8),
        dilation=1
    )
    # update dimensions
    active_height = do_conv2d_dim_math(active_height, model_skeleton["conv_1"], 0)
    active_width = do_conv2d_dim_math(active_width, model_skeleton["conv_1"], 1)
    print((active_height, active_width, solve_pad(4, img_height, 8)))

    model_skeleton["relu_1"] = nn.ReLU()

    ######################################################
    # second convolution
    model_skeleton["conv_2"] = nn.Conv2d(
        in_channels=32,
        out_channels=64,
        kernel_size=4,
        stride=2,
        padding=solve_pad(2, img_height, 4),
        dilation=1
    )
    # update dims
    active_height = do_conv2d_dim_math(active_height, model_skeleton["conv_2"], 0)
    active_width = do_conv2d_dim_math(active_width, model_skeleton["conv_2"], 1)
    print((active_height, active_width, solve_pad(2, img_height, 4)))

    model_skeleton["relu_2"] = nn.ReLU()


    ######################################################
    # third convolution
    model_skeleton["conv_3"] = nn.Conv2d(
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=solve_pad(1, img_height, 3),
        dilation=1
    )

    active_height = do_conv2d_dim_math(active_height, model_skeleton["conv_3"], 0)
    active_width = do_conv2d_dim_math(active_width, model_skeleton["conv_3"], 1)
    print((active_height, active_width, solve_pad(1, img_height, 3)))

    model_skeleton["relu_3"] = nn.ReLU()

    ######################################################
    # Linear layers
    model_skeleton["flattener"] = nn.Flatten() # I think this is the analog?
    model_skeleton["fc_1"] = nn.Linear((64*img_height*img_width-3+2)+1, 512)
    # model_skeleton["fc_1"] = nn.Linear(64 * active_height * active_width, 512)
    model_skeleton["relu_4"] = nn.ReLU()
    model_skeleton["fc_out"] = nn.Linear(512, num_actions)

    ### all done
    return model_skeleton

### NOTE: THE AUTOGRADER DOES NOT ACCEPT THIS, BUT LEAVING HERE FOR KICKS AND GIGGLES.
class QApproxNet(nn.Module):
    """
    My only experience with torch involves setting it up this way, so making a class here for each of the networks
    template grabbed here: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    """
    def __init__(self, state_shape, num_actions, config):
        """
        To simplify, we specify the paddings as:
            (stride - 1) * img_height - stride + filter_size) // 2
        """
        super(QApproxNet, self).__init__()
        # save architecture params
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.config = config
        # have on hand
        img_height, img_width, n_channels = state_shape
        # for tracking the sizes of the images throughout
        active_height, active_width = img_height, img_width
        print((active_height, active_width))

        ######################################################
        # first convolution
        self.conv_1 = nn.Conv2d(
            in_channels=n_channels * self.config.state_history, # use the lag-4 channel structure
            out_channels=32, # first layer, per paper
            kernel_size=8,# first layer, per paper
            stride=4, # first layer, per paper
            padding=solve_pad(4, img_height, 8),
            dilation=1
        )
        # update dimensions
        active_height = do_conv2d_dim_math(active_height, self.conv_1, 0)
        active_width = do_conv2d_dim_math(active_width, self.conv_1, 1)
        print((active_height, active_width, solve_pad(4, img_height, 8)))

        ######################################################
        # second convolution
        self.conv_2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=solve_pad(2, img_height, 4),
            dilation=1
        )
        # update dims
        active_height = do_conv2d_dim_math(active_height, self.conv_2, 0)
        active_width = do_conv2d_dim_math(active_width, self.conv_2, 1)
        print((active_height, active_width, solve_pad(2, img_height, 4)))


        ######################################################
        # third convolution
        self.conv_3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=solve_pad(1, img_height, 3),
            dilation=1
        )
        active_height = do_conv2d_dim_math(active_height, self.conv_3, 0)
        active_width = do_conv2d_dim_math(active_width, self.conv_3, 1)
        print((active_height, active_width, solve_pad(1, img_height, 3)))

        ######################################################
        # Linear layers
        self.fc_1 = nn.Linear(64 * active_height * active_width, 512)
        self.fc_out = nn.Linear(512, self.num_actions)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = F.relu(self.fc_1(x))
        x = self.fc_out(x)
        return x


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    Model configuration can be found in the Methods section of the above paper.
    """

    def transfer_weights(self):
        """Helper for weight transfer"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def initialize_models(self):
        """Creates the 2 separate networks (Q network and Target network). The input
        to these models will be an img_height * img_width image
        with channels = n_channels * self.config.state_history

        1. Set self.q_network to be a model with num_actions as the output size
        2. Set self.target_network to be the same configuration self.q_network but initialized from scratch
        3. What is the input size of the model?

        To simplify, we specify the paddings as:
            (stride - 1) * img_height - stride + filter_size) // 2

        Hints:
            1. Simply setting self.target_network = self.q_network is incorrect.
            2. The following functions might be useful
                - nn.Sequential
                - nn.Conv2d
                - nn.ReLU
                - nn.Flatten
                - nn.Linear
        """
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n

        ##############################################################
        ################ YOUR CODE HERE - 25-30 lines lines ################

        print("------ Environment Overview ------")
        print(f"state_shape: {state_shape}")
        print(f"num_actions: {num_actions}")

        # self.q_network = QApproxNet(
        #     state_shape=state_shape,
        #     num_actions=num_actions,
        #     config=self.config
        # )

        # self.target_network = QApproxNet(
        #     state_shape=state_shape,
        #     num_actions=num_actions,
        #     config=self.config
        # )


        self.q_network = nn.Sequential(
            make_dqn_skeleton(
                state_shape=state_shape,
                num_actions=num_actions,
                config=self.config
            )
        )

        self.target_network = nn.Sequential(
            make_dqn_skeleton(
                state_shape=state_shape,
                num_actions=num_actions,
                config=self.config
            )
        )

        print("------ Model Architecture ------")
        print(f"QNet:")
        print(self.q_network)
        print(f"TargNet:")
        print(self.target_network)

        self.transfer_weights()

        ##############################################################
        ######################## END YOUR CODE #######################

    def get_q_values(self, state, network):
        """
        Returns Q values for all actions

        Args:
            state: (torch tensor)
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            network: (str)
                The name of the network, either "q_network" or "target_network"

        Returns:
            out: (torch tensor) of shape = (batch_size, num_actions)

        Hint:
            1. What are the input shapes to the network as compared to the "state" argument?
            2. You can forward a tensor through a network by simply calling it (i.e. network(tensor))
        """
        ##############################################################
        ################ YOUR CODE HERE - 4-5 lines lines ################

        assert network in ("q_network", "target_network")
        state_ = state.transpose_(1, 3) # currently channels on last axis, need to flip
        out = self.q_network(state_) if network == "q_network" else self.target_network(state_)

        # print("------ Shape Check ------")
        # print(f"network: {network}")
        # print(f"state_: {state.shape}")
        # print(f"state_: {state_.shape}")
        # print(f"num_actions: {out.shape}")
        ##############################################################
        ######################## END YOUR CODE #######################
        return out


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((8, 8, 6))
    # env = EnvTest((5, 5, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
