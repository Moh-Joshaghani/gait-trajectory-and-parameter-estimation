"""
Tools and classes that will be used to build the models
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import models as torch_models

torch.set_default_dtype(torch.float)

# ===================================================================================
#                   FUNCTIONS

def act_fun(act_type):
    """
    returns the activation funciton based on the type
    :param act_type:
    :return:
    """

    if act_type == "relu":
        return nn.ReLU()

    if act_type == "tanh":
        return nn.Tanh()

    if act_type == "sigmoid":
        return nn.Sigmoid()

    if act_type == "leaky_relu":
        return nn.LeakyReLU(0.2)

    if act_type == "elu":
        return nn.ELU()

    if act_type == "softmax":
        return nn.LogSoftmax(dim=1)

    if act_type == "linear":
        # # return nn.LeakyReLU(1)  # initialized like this, but not used in forward!
        return linear_act_function()


# ===================================================================================
# ===================================================================================
def build_core_network(in_dim, opt):
    """
    builds the core part of the network, based on the value in opt.net_type
    :return: core_network
    """
    if opt.net_type == 'DNN':
        MLP1_arch = {'input_dim': in_dim,
                     'fc_lay':  opt.fc_lay,
                     'fc_drop':  opt.fc_drop,
                     'fc_use_batchnorm':  opt.fc_use_batchnorm,
                     'fc_use_laynorm':  opt.fc_use_laynorm,
                     'fc_use_laynorm_inp':  opt.fc_use_laynorm_inp,
                     'fc_use_batchnorm_inp':  opt.fc_use_batchnorm_inp,
                     'fc_act':  opt.fc_act,
                     }

        core_network = MLP(MLP1_arch)

    # identity
    elif opt.net_type == 'identity':
        core_network = nn.Identity()
        core_network.out_dim = in_dim

    # resnet 34
    elif opt.net_type == 'resnet34':
        core_network = torch_models.resnet34(pretrained=False)
        core_network = build_core_network_from_resnet(net=core_network, remove_first=True)
        core_network.out_dim = 512

    # resnet 18
    elif opt.net_type == 'resnet18':
        core_network = torch_models.resnet18(pretrained=False)
        core_network = build_core_network_from_resnet(net=core_network, remove_first=True)
        core_network.out_dim = 512


    # resnet 50
    elif opt.net_type == 'resnet50':
        core_network = torch_models.resnet50(pretrained=False)
        core_network = build_core_network_from_resnet(net=core_network, remove_first=True)
        core_network.out_dim = 2048

    elif opt.net_type == 'resnet101':
        core_network = torch_models.resnet101(pretrained=False)
        core_network = build_core_network_from_resnet(net=core_network, remove_first=True)
        core_network.out_dim = 2048

    return core_network


# ====================================================================================
# ====================================================================================
def build_core_network_from_resnet(net, remove_first=True):
    """
    removes the extra layers from resnet architecture
    :param net: resnet
    :param remove_first: if remove the first layer or not
    :return:
    """
    layers = []
    named_children = net.named_children()
    net_params = list(net.parameters())

    # adding the first layers before layer0, or not
    if not remove_first:
        for lay_name, lay in named_children:
            #
            if 'layer' in lay_name:
                break

            layers.append(lay)

    # adding the 'layer' and average pool parts to the network
    for lay_name, lay in named_children:
        if 'layer' in lay_name or 'avgpool' in lay_name:
        # if 'layer' in lay_name:
            layers.append(lay)


    core_net = nn.Sequential(*layers)

    return core_net



#                           CLASSES
# =====================================================================================
# =====================================================================================
class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

# =====================================================================================
# =====================================================================================
class LayerNorm(nn.Module):
    """
    layer norm class
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # # print(np.shape(x))
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


# =====================================================================================
# =====================================================================================
class MLP(nn.Module):
    def __init__(self, options, flatten=True):
        """
        inits, the architecture as a dictionary must be passed!
        :param options: the dictionary of the architecture
        """
        super(MLP, self).__init__()

        self.input_dim = int(options['input_dim'])
        self.fc_lay = options['fc_lay']
        self.fc_drop = options['fc_drop']
        self.fc_use_batchnorm = options['fc_use_batchnorm']
        self.fc_use_laynorm = options['fc_use_laynorm']
        self.fc_use_laynorm_inp = options['fc_use_laynorm_inp']
        self.fc_use_batchnorm_inp = options['fc_use_batchnorm_inp']
        self.fc_act = options['fc_act']

        self.wx = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        self.flatten = flatten

        # input layer normalization
        if self.fc_use_laynorm_inp:
            self.ln0 = LayerNorm((self.input_dim))

        # input batch normalization
        if self.fc_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d([self.input_dim], momentum=0.05)

        self.N_fc_lay = len(self.fc_lay)

        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_fc_lay):

            # dropout
            self.drop.append(nn.Dropout(p=self.fc_drop[i]))

            # activation
            self.act.append(act_fun(self.fc_act[i]))

            add_bias = True

            # layer norm initialization
            self.ln.append(LayerNorm((self.fc_lay[i])))
            self.bn.append(nn.BatchNorm1d(self.fc_lay[i], momentum=0.05))

            if self.fc_use_laynorm[i] or self.fc_use_batchnorm[i]:
                add_bias = False

            # Linear operations
            self.wx.append(nn.Linear(current_input, self.fc_lay[i], bias=add_bias))

            # weight initialization
            self.wx[i].weight = torch.nn.Parameter(
                torch.Tensor(self.fc_lay[i], current_input).uniform_(-np.sqrt(0.01 / (current_input + self.fc_lay[i])),
                                                                     np.sqrt(0.01 / (current_input + self.fc_lay[i]))))
            self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.fc_lay[i]))

            current_input = self.fc_lay[i]


    # ====================================================================================
    # ====================================================================================
    def forward(self, x):

        # flattening, if required
        if self.flatten:
            x = Flatten()(x)

        # Applying Layer/Batch Norm
        if bool(self.fc_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.fc_use_batchnorm_inp):
            x = self.bn0((x))

        for i in range(self.N_fc_lay):
            # print('input_shape:{}'.format(x.shape))

            # # # if self.fc_act[i] != 'linear':

            if self.fc_use_laynorm[i]:
                x = self.drop[i](self.act[i](self.ln[i](self.wx[i](x))))
                # x = self.drop[i](self.ln[i](self.act[i](self.wx[i](x))))

            if self.fc_use_batchnorm[i]:
                x = self.drop[i](self.act[i](self.bn[i](self.wx[i](x))))
                # x = self.drop[i](self.bn[i](self.act[i](self.wx[i](x))))

            if self.fc_use_batchnorm[i] == False and self.fc_use_laynorm[i] == False:
                x = self.drop[i](self.act[i](self.wx[i](x)))

        return x


# =====================================================================================
# =====================================================================================
class linear_act_function(nn.Module):
    def __init__(self):
        super(linear_act_function, self).__init__()

    def forward(self, x):
        return x


# =====================================================================================
# =====================================================================================
class CNN(nn.Module):

    def __init__(self, options):
        super(CNN, self).__init__()

        self.cnn_N_filt = options['cnn_N_filt']
        self.cnn_len_filt = options['cnn_len_filt']
        self.cnn_max_pool_len = options['cnn_max_pool_len']
        self.num_data_channels = 6

        self.cnn_act = options['cnn_act']
        self.cnn_drop = options['cnn_drop']

        self.cnn_use_laynorm = options['cnn_use_laynorm']
        self.cnn_use_batchnorm = options['cnn_use_batchnorm']
        self.cnn_use_laynorm_inp = options['cnn_use_laynorm_inp']
        self.cnn_use_batchnorm_inp = options['cnn_use_batchnorm_inp']

        self.input_dim = int(options['input_dim'])

        self.N_cnn_lay = len(options['cnn_N_filt'])

        self.conv = nn.ModuleList([])
        self.conv.append(nn.Conv2d(self.num_data_channels, self.cnn_N_filt[0], (1, self.cnn_len_filt[0]), groups=1))

        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        if self.cnn_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        if self.cnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm2d([1, self.input_dim], momentum=0.05)

        current_input = self.input_dim

        for i in range(self.N_cnn_lay):

            N_filt = int(self.cnn_N_filt[i])
            len_filt = int(self.cnn_len_filt[i])

            # dropout
            self.drop.append(nn.Dropout(p=self.cnn_drop[i]))

            # activation
            self.act.append(act_fun(self.cnn_act[i]))

            # layer norm initialization
            self.ln.append(
                LayerNorm([N_filt, 1, int((current_input - len_filt + 1) / self.cnn_max_pool_len[i])]))

            self.bn.append(
                nn.BatchNorm2d(N_filt, int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i]),
                               momentum=0.05))

            if i > 0:
                self.conv.append(nn.Conv2d(self.cnn_N_filt[i - 1], self.cnn_N_filt[i], (1, self.cnn_len_filt[i]), bias=True))

            current_input = int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i])

        self.out_dim = current_input * N_filt


    # ============================================================================================
    # ============================================================================================
    def forward(self, x):
        """
        forward part
        :param x:
        :return:
        """
        batch = x.shape[0]
        num_channels = x.shape[1]
        try:
            seq_len = x.shape[2]
        except:
            a = 1

        if bool(self.cnn_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.cnn_use_batchnorm_inp):
            x = self.bn0((x))


        # print('CNN layer')
        for i in range(self.N_cnn_lay):
            # print('input_shape:{}'.format(x.shape))
            if self.cnn_use_laynorm[i]:
                if i == 0:

                    x = self.drop[i](
                        self.act[i](self.ln[i](F.max_pool2d(torch.abs(self.conv[i](x)), (1, self.cnn_max_pool_len[i])))))
                        # self.ln[i](self.act[i](F.max_pool2d(torch.abs(self.conv[i](x)), (1, self.cnn_max_pool_len[i])))))
                else:
                    x = self.drop[i](self.act[i](self.ln[i](F.max_pool2d(self.conv[i](x), (1, self.cnn_max_pool_len[i])))))
                    # x = self.drop[i](self.ln[i](self.act[i](F.max_pool2d(self.conv[i](x), (1, self.cnn_max_pool_len[i])))))

            if self.cnn_use_batchnorm[i]:
                x = self.drop[i](self.bn[i](self.act[i](F.max_pool2d(self.conv[i](x), (1, self.cnn_max_pool_len[i])))))

            if self.cnn_use_batchnorm[i] == False and self.cnn_use_laynorm[i] == False:
                x = self.drop[i](self.act[i](F.max_pool2d(self.conv[i](x), (1, self.cnn_max_pool_len[i]))))


        return x


