"""
This module includes graphical tools to visualize the network and the data
"""
import matplotlib.pyplot as plt
import numpy as np
import os

#   METHODS
def visualize_label_distribution():
    pass


# ===========================================================================================
def plot_labels_vs_predicted_class(sample_outputs, sample_labels, label_names, save_name):
    """
    Plots predicted vs output values and saves the plot. helps to visualize network predicted
    values for different input
    :param sample_outputs: list of numpy
    :param sample_labels: list of numpy
    :param label_names: list
    :param save_name: full directory of save
    :return:
    """

    # plotting some sample input and output predicted values for each
    fig, axs = plt.subplots(np.shape(sample_outputs[0])[-1], 1, figsize=(15, 10))
    plt.title(save_name.split("/")[-1].split(".")[0])

    t = range(len(sample_outputs))

    if not type(axs) == list:
        axs = [axs]

    # sketching the subplots
    for label_idx in range(np.shape(sample_outputs[0])[-1]):
        axs[label_idx].scatter(t, [s[0][label_idx] for s in sample_outputs], c='r', marker='X', s=10, label='Predicted')
        axs[label_idx].scatter(t, [s[0][label_idx] for s in sample_labels], c='b', marker='o', s=14,
                               label='Ground truth')
        axs[label_idx].scatter(t, [s[0][label_idx] * 1.05 for s in sample_labels], c='g', marker='_', label='Upper 5%')
        axs[label_idx].scatter(t, [s[0][label_idx] * 0.95 for s in sample_labels], c='g', marker='_', label='Lower 5%')
        axs[label_idx].legend()
        axs[label_idx].set_title('{}'.format(label_names[label_idx]))

    plt.savefig(save_name)
    plt.close()

# ===========================================================================================
def error_variance_history_bar_plot(valid_mae_each, valid_std_each, epoch, save_name):
    """
    plotting the error and variance of error for each of the predicted values
    :param valid_mae_each:
    :param valid_std_each:
    :param epoch:
    :param save_name:
    :return:
    """

    plt.figure(figsize=(16, 10), dpi=120)
    plt.ioff()
    num_plots = np.shape(valid_mae_each)[-1]

    # plotting error of each of the target variables
    for i in range(num_plots):
        plt.subplot(num_plots, 1, i + 1)
        t = np.linspace(1, len(valid_mae_each), len(valid_mae_each), dtype=int)
        plt.errorbar(t, valid_mae_each[:, i], valid_std_each[:, i], linestyle='None', marker='^')
        plt.title('Error and std in {}-th epoch'.format(epoch))
        plt.grid(True)

    plt.savefig(save_name)
    plt.close()

# ===========================================================================================
def plot_loss(loss_total_train, validation_loss_total, save_frequency, save_name):
    
    plt.figure()
    t = np.linspace(1, len(loss_total_train) * save_frequency, len(loss_total_train), dtype=int)
    plt.plot(t, loss_total_train, '*-', label='Train loss')
    plt.plot(t, validation_loss_total, '*-', label='Validation loss')
    plt.grid(True)

    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss value')
    plt.title('Loss values over training')
    plt.legend()
    plt.savefig(save_name)
    plt.close()

# ======================================================================================
def visualize_cnn_kernels(cnn_conv, save_dir):
    """
    plots the cnn kernels as an image
    :param cnn_conv:
    :param save_name:
    :return:
    """

    for idx, conv_layer in enumerate(cnn_conv):
        cnn_weights = conv_layer.weight.cpu().detach().numpy()
        cnn_weights = np.reshape(cnn_weights, (-1, np.shape(cnn_weights)[0]))
        plt.figure(figsize=(25, 25))

        plt.autoscale()
        plt.imshow(cnn_weights)

        plt.savefig(os.path.join(save_dir, 'kernels_layer_{}.png'.format(idx)), bbox_inches='tight', dpi=200)
        plt.close()

# ==========================================================================================
def bland_altman_plot(valid_output, valid_label, label_names, epoch, save_name):
    """
    bland altman plot for 2-d data N_data * dim_output
    :param valid_output: list of numpy
    :param valid_label:list of numpy
    :param label_names: name of the predicted values
    :param epoch:
    :param save_name: save name
    :return:
    """
    plt.figure(figsize=(16, 20))
    plt.suptitle("Bland-Altman plot in {} th epoch".format(epoch))
    num_plots = np.shape(valid_output)[-1]

    for i in range(num_plots):
        plt.subplot(num_plots, 1, i+1)
        mean = (valid_label[:, i] + valid_output[:, i])/2
        diff = valid_label[:, i] - valid_output[:, i]  # Difference between data1 and data2
        md = np.mean(diff)  # Mean of the difference
        sd = np.std(diff, axis=0)  # Standard deviation of the difference

        plt.scatter(mean, diff)
        plt.axhline(md, color='gray', linestyle='--')
        plt.axhline(md + 1.98 * sd, color='gray', linestyle='--')
        plt.axhline(md - 1.98 * sd, color='gray', linestyle='--')

        plt.suptitle('{}'.format(label_names[i]))
        plt.xlabel('Mean')
        plt.ylabel('Difference')

    plt.savefig(save_name)
    plt.close()


# ====================================================================================
# ====================================================================================
def trajectory_vs_predicted(sample_outputs, sample_labels, label_names, save_name):

    fig, axs = plt.subplots(len(label_names), 1, figsize=(15, 10))
    t = range(np.shape(sample_outputs)[-1])

    for label_idx in range(len(label_names)):
        axs.plot(t, sample_outputs[0][0], c='r', ms=4, label='Predicted')
        axs.plot(t, sample_labels[0][0][0], c='b', ms=3, label='Ground truth')
        axs.plot(t, 1.05 * np.asarray(sample_labels[0][0][0]), c='g', marker='_', ms=2, label='Upper 5%')
        axs.plot(t, 0.95 * np.asarray(sample_labels[0][0][0]), c='g', marker='_', ms=2, label='Lower 5%')
        axs.legend()
        axs.set_title('{}'.format(label_names[label_idx]))

    plt.savefig(save_name)
    plt.close()


# ======================================================================================
# ======================================================================================
def trajectory_bland_altman(valid_output, valid_label, label_names, epoch, save_name):
    plt.figure(figsize=(16, 20))
    plt.suptitle("Bland-Altman plot in {} th epoch".format(epoch))

    num_plots = len(label_names)
    for i in range(num_plots):
        plt.subplot(num_plots, 1, i + 1)
        mean = np.mean((valid_label + valid_output) / 2, axis=1)
        diff = np.mean(valid_label - valid_output, axis=1)  # Difference between data1 and data2
        md = np.mean(diff)  # Mean of the difference
        sd = np.std(diff, axis=0)  # Standard deviation of the difference

        plt.scatter(mean, diff)
        plt.axhline(md, color='gray', ms=4, linestyle='-.', label='mean')
        plt.axhline(md + 1.96 * sd, ms=5, color='gray', linestyle='--', label='1.96 std')
        plt.axhline(md - 1.96 * sd, ms=5, color='gray', linestyle='--')

        plt.title('{}'.format(label_names[i]))
        plt.xlabel('Mean')
        plt.ylabel('Difference')

    plt.savefig(save_name)
    plt.close()









