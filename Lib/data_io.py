"""
This module reads the config file
"""

import configparser as ConfigParser
from optparse import OptionParser
import numpy as np
# import scipy.io.wavfile
import torch



# =====================================================================================
# =====================================================================================
def read_conf_inp_raw(cfg_file):
    """
    Reads the config from a external config file .cfg
    :param cfg_file:
    :return:
    """
    parser = OptionParser()
    (options, args) = parser.parse_args()

    Config = ConfigParser.ConfigParser()
    Config.read(cfg_file)

    # input
    options.cw_len = Config.get('input', 'cw_len')
    options.batch_size = Config.get('input', 'batch_size')
    options.sensor_location = Config.get('input', 'sensor_location')

    # [simulation]
    options.aug = Config.get('simulation', 'Augmentation')
    options.num_aug = Config.get('simulation', 'num_aug')
    options.estimate_trajectory = Config.get('simulation', 'estimate_trajectory')
    options.cost_function = Config.get('simulation', 'cost_function')

    # [cnn]
    options.cnn_N_filt = Config.get('cnn', 'cnn_N_filt')
    options.cnn_len_filt = Config.get('cnn', 'cnn_len_filt')
    options.cnn_max_pool_len = Config.get('cnn', 'cnn_max_pool_len')
    options.cnn_use_laynorm_inp = Config.get('cnn', 'cnn_use_laynorm_inp')
    options.cnn_use_batchnorm_inp = Config.get('cnn', 'cnn_use_batchnorm_inp')
    options.cnn_use_laynorm = Config.get('cnn', 'cnn_use_laynorm')
    options.cnn_use_batchnorm = Config.get('cnn', 'cnn_use_batchnorm')
    options.cnn_act = Config.get('cnn', 'cnn_act')
    options.cnn_drop = Config.get('cnn', 'cnn_drop')
    options.pt = Config.get('cnn', 'pt')
    options.cnn_freeze = Config.get('cnn', 'cnn_freeze')

    # [core_network]
    options.net_type = Config.get('core_network', 'net_type')


    # [regressir]
    options.fc_lay = Config.get('regressor', 'fc_lay')
    options.fc_drop = Config.get('regressor', 'fc_drop')
    options.fc_use_laynorm_inp = Config.get('regressor', 'fc_use_laynorm_inp')
    options.fc_use_batchnorm_inp = Config.get('regressor', 'fc_use_batchnorm_inp')
    options.fc_use_batchnorm = Config.get('regressor', 'fc_use_batchnorm')
    options.fc_use_laynorm = Config.get('regressor', 'fc_use_laynorm')
    options.fc_act = Config.get('regressor', 'fc_act')


    return options



# ======================================================================================
# ======================================================================================
def read_conf_inp(cfg_file):
    """
    reads out the config from outer .cfg file
    :param raw_opt:
    :return:
    """

    # reading the config file
    raw_opt = read_conf_inp_raw(cfg_file)

    parser = OptionParser()
    (opt, args) = parser.parse_args()


    # [input]
    opt.cw_len = int(raw_opt.cw_len)
    opt.batch_size = int(raw_opt.batch_size)
    opt.sensor_location = raw_opt.sensor_location

    # [simulation]
    opt.aug = str_to_bool(raw_opt.aug)
    opt.estimate_trajectory = str_to_bool(raw_opt.estimate_trajectory)
    opt.num_aug = int(raw_opt.num_aug)
    opt.cost_function = raw_opt.cost_function

    # [cnn]
    opt.cnn_N_filt = list(map(int, raw_opt.cnn_N_filt.split(',')))
    opt.cnn_len_filt = list(map(int, raw_opt.cnn_len_filt.split(',')))
    opt.cnn_max_pool_len = list(map(int, raw_opt.cnn_max_pool_len.split(',')))
    opt.cnn_use_laynorm_inp = str_to_bool(raw_opt.cnn_use_laynorm_inp)
    opt.cnn_use_batchnorm_inp = str_to_bool(raw_opt.cnn_use_batchnorm_inp)
    opt.cnn_use_laynorm = list(map(str_to_bool, raw_opt.cnn_use_laynorm.split(',')))
    opt.cnn_use_batchnorm = list(map(str_to_bool, raw_opt.cnn_use_batchnorm.split(',')))
    opt.cnn_act = list(map(str, raw_opt.cnn_act.split(',')))
    opt.cnn_drop = list(map(float, raw_opt.cnn_drop.split(',')))
    opt.cnn_pt = raw_opt.pt
    opt.cnn_freeze = raw_opt.cnn_freeze

    # core network
    opt.net_type = raw_opt.net_type

    # [dnn]
    opt.fc_lay = list(map(int, raw_opt.fc_lay.split(',')))
    opt.fc_drop = list(map(float, raw_opt.fc_drop.split(',')))
    opt.fc_use_laynorm_inp = str_to_bool(raw_opt.fc_use_laynorm_inp)
    opt.fc_use_batchnorm_inp = str_to_bool(raw_opt.fc_use_batchnorm_inp)
    opt.fc_use_batchnorm = list(map(str_to_bool, raw_opt.fc_use_batchnorm.split(',')))
    opt.fc_use_laynorm = list(map(str_to_bool, raw_opt.fc_use_laynorm.split(',')))
    opt.fc_act = list(map(str, raw_opt.fc_act.split(',')))

    return opt

# ===================================================================================================
# ===================================================================================================
def str_to_bool(inp):
    """
    converts true and false to bool
    :param inp:
    :return:
    """

    return inp == 'True' or inp == 'true'
