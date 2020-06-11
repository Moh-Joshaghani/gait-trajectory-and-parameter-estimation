"""
This module loads a pretrained model and manipulate the last layers
"""

import torch
import torch.nn as nn

from Lib import model_tools as mt

torch.set_default_dtype(torch.float)


# =================================================================================
# =================================================================================
class my_model(nn.Module):
    def __init__(self, input_dim, output_dim, opt, device='cpu'):
        """

        :param input_dim:
        :param output_dim:
        :param opt:
        """

        super(my_model, self).__init__()

        # reading dimensions
        self.input_dim = input_dim  # -1, num channels, len
        self.num_data_channel = input_dim[1]
        self.len_data = input_dim[2]
        self.output_dim = output_dim

        # reading the options
        self.opt = opt

        # device
        self.device = device

        # making the model!
        self.CNN, self.core_network, self.Regressor = self.__build_model_from_scratch()
        #

    # ====================================================================================
    def __build_model_from_scratch(self):
        """
        building the model from scratch
        :return:
        """

        # rebuilding the convolution architecture based on the overall option
        # The first par: the CNN part
        CNN_arch = {'input_dim': self.opt.cw_len,
                    'cnn_N_filt': self.opt.cnn_N_filt,
                    'cnn_len_filt': self.opt.cnn_len_filt,
                    'cnn_max_pool_len': self.opt.cnn_max_pool_len,
                    'cnn_use_laynorm_inp': self.opt.cnn_use_laynorm_inp,
                    'cnn_use_batchnorm_inp': self.opt.cnn_use_batchnorm_inp,
                    'cnn_use_laynorm': self.opt.cnn_use_laynorm,
                    'cnn_use_batchnorm': self.opt.cnn_use_batchnorm,
                    'cnn_act': self.opt.cnn_act,
                    'cnn_drop': self.opt.cnn_drop,
                    }

        CNN_net = mt.CNN(CNN_arch)
        CNN_net.to(self.device)


        # The core Network:
        core_net = mt.build_core_network(CNN_net.out_dim, self.opt)
        core_net.to(self.device)

        MLP2_arch = {'input_dim': core_net.out_dim,
                     'fc_lay': self.opt.fc_lay + [self.output_dim[-1]],
                     'fc_drop': self.opt.fc_drop + [0.0],
                     'fc_use_batchnorm': self.opt.fc_use_batchnorm + [False],
                     'fc_use_laynorm': self.opt.fc_use_laynorm + [False],
                     'fc_use_laynorm_inp': self.opt.fc_use_laynorm_inp,
                     'fc_use_batchnorm_inp': self.opt.fc_use_batchnorm_inp,
                     'fc_act': self.opt.fc_act + ['linear'],
                     }

        regressor_net = mt.MLP(MLP2_arch)
        regressor_net.to(self.device)



        return [CNN_net, core_net, regressor_net]

