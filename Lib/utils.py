import os
import datetime
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def timestamp():
    """
    Computes and returns current timestamp

    Args:
    -----
    timestamp: String
        Current timestamp in formate: hh-mm-ss
    """

    timestamp = str(datetime.datetime.now()).split('.')[0].replace(' ', '_').replace(':', '-')
    return timestamp


def create_directory(path):
    """
    Method that creates a directory if it does not already exist

    Args
    ----
    path: String
        path to the directory to be created
    dir_existed: boolean
        Flag that captures of the directory existed or was created
    """
    dir_existed = True
    if not os.path.exists(path):
        os.makedirs(path)
        dir_existed = False
    return dir_existed


# ======================================================================================
# ======================================================================================
def get_loss_stats(loss_list):
    """
    Computes loss statistics given a list of loss values

    FOR FURTHER PROCESSING

    Args:
    -----
    loss_list: List
        List containing several loss values
    """

    if(len(loss_list)==0):
        return

    loss_np = torch.stack(loss_list)
    avg_loss = torch.mean(loss_np)
    max_loss = torch.max(loss_np)
    min_loss = torch.min(loss_np)

    return avg_loss


# =====================================================================================
# =====================================================================================
def plot_two_graphs_together(train_loss, val_loss, sav_freq, save_name):
    plt.figure()
    t = np.linspace(1, len(train_loss) * sav_freq, len(train_loss), dtype=int)
    plt.plot(t, train_loss, '*-', label='train loss')
    plt.plot(t, val_loss, '*-', label='validation loss')

    plt.legend()
    plt.savefig(save_name)

    return



# =====================================================================================
# =====================================================================================
def analyse_data_set(data_loader, save_dir):
    """
     calculating the subject data count, mean, and variance in an exhaustive way to avoid loading
     big data into the RAM
    :param data_loader: the data loader generator object
    :param save_dir: save dir in which the data is saved
    :return:
    """
    count_dict = dict()
    label_dict = dict()


    #
    for subj, trial, step in data_loader.sampler.indices:

        # update the count
        if subj in count_dict.keys():
            count_dict[subj] += 1

        else:
            count_dict.update({subj: 1})


    # plotting the distribution
    plt.figure()
    plt.bar(count_dict.keys(), count_dict.values())
    plt.title('amount of data available for each subject')

    plt.savefig(save_dir)
    plt.close()


# ==========================================================================================
# ==========================================================================================
def get_label_distribution(train_loader, valid_loader, plot_save_dir):
    """
    This method returns the distribution of the labels
    :param data_loader:
    :return:
    """
    # reading the labels in train data
    train_labels = []
    for i, (x, labels) in enumerate(train_loader):
        try:
            train_labels.extend(labels.squeeze().tolist())
        except:
            train_labels.append(labels.squeeze().tolist())


    # reading the labels in test data
    valid_labels = []
    for i, (x, labels) in enumerate(valid_loader):
        try:
            valid_labels.extend(labels.squeeze().tolist())
        except:
            valid_labels.append(labels.squeeze().tolist())


    # plotting the histogram only for train data
    plt.figure(figsize=(15, 10))
    plt.subplot(311)
    # train labels individually
    bins = np.linspace(np.min(train_labels), np.max(train_labels), 100)
    bins_values, _ = np.histogram(train_labels, bins=bins, density=True)
    plt.grid()
    plt.title('histogram of the labels')
    plt.plot(bins[:-1], bins_values)

    # test labels individually
    plt.subplot(312)
    bins = np.linspace(np.min(valid_labels), np.max(valid_labels), 100)
    bins_values, _ = np.histogram(valid_labels, bins=bins, density=True)
    plt.grid()
    plt.title('histogram of the labels')
    plt.plot(bins[:-1], bins_values)

    # all labels together
    plt.subplot(313)
    bins = np.linspace(np.min(valid_labels + train_labels), np.max(valid_labels + train_labels), 100)
    valid_bins, _ = np.histogram(valid_labels, bins=bins, density=True)
    train_bins, _ = np.histogram(train_labels, bins=bins, density=True)

    plt.grid()
    plt.title('histogram of the labels')
    plt.plot(bins[:-1], valid_bins, label='test')
    plt.plot(bins[:-1], train_bins, label='train')

    plt.legend()
    plt.savefig(os.path.join(plot_save_dir, 'dataset_distribution.png'))











