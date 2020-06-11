import os

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.utils.data as data

import Lib.utils as utils
import Models.models as models
import Lib.DataLoader as Datasets
from Lib.data_io import read_conf_inp
from Lib import graphical_tools as gt
from tqdm import tqdm
from shutil import copyfile
from random import sample



class Trainer:
    def __init__(self, cfg_file):
        """
        Initializing trainer
        """

        # training parameters
        self.learning_rate = 0.001
        self.valid_batch_size = 32


        self.max_epochs = 200
        self.valid_size = 0.2
        self.save_frequency = 5
        self.use_lr_scheduler = False
        self.scheduler = []

        # reading the setting
        self.opt = read_conf_inp(cfg_file)

        self.location = self.opt.sensor_location  # 'Cavity'  #

        # output labels:
        self.step_label_tag = []  # ['Sole_Angle', 'Impact_Angle']
        self.stride_label_tag = ['Stridelength_FF']
        self.trajectory_labels = ['loc_y']  # ['loc_x', 'loc_y', 'loc_z']
        # self.trajectory_labels = ['orientation_x', 'orientation_y', 'orientation_z', 'loc_x', 'loc_y',
        #                           'loc_z']

        if self.opt.estimate_trajectory:
            self.stride_label_tag = []
            self.step_label_tag = []

        #  Autmentation
        self.enable_augmentation = self.opt.aug
        self.num_roll_augmentation = self.opt.num_aug
        self.batch_size = self.opt.batch_size

        # network dims
        self.input_dim = (-1, 6, self.opt.cw_len)

        if self.opt.estimate_trajectory:
            self.output_dim = (-1, self.input_dim[-1])
            self.label_names = self.trajectory_labels

        else:
            # output dim: depending if stride labels or step labels is used
            self.output_dim = (-1, len(self.step_label_tag) + len(self.stride_label_tag))
            self.label_names = self.step_label_tag + self.stride_label_tag

        # Network settings (will be re-initiated outside init)
        self.CNN_optimizer = None
        self.core_optimizer = None
        self.regressor_optimizer = None

        # loss functions
        self.loss_function = None

        # accuracy measures
        self.acc_bound_rel = 0.1
        self.train_accuracy = []
        self.valid_accuracy = []

        self.validation_loss_total = []
        self.best_validation_loss = 1e15
        self.loss_total_train = []
        self.train_mse = []

        self.all_eval_mse_error = []
        self.all_eval_variances = []

        self.valid_mae_each = np.zeros((0, len(self.label_names)))
        self.valid_std_each = np.zeros((0, len(self.label_names)))

        self.valid_output = 0
        self.valid_label = 0



        # relevant directories
        self.root = os.getcwd()
        self.config_file = cfg_file
        self.data_path = os.path.join(self.root, "data/Vicon_running_data_set")

        if self.opt.estimate_trajectory:
            self.output_path = os.path.join(os.getcwd(), "trajectory_experiments", utils.timestamp())
            self.num_sample_test = 1

        else:
            self.output_path = os.path.join(os.getcwd(), "parameter_experiments", utils.timestamp())
            self.num_sample_test = 50

        self.model_path = os.path.join(self.output_path, 'models')


        # sample data for visualization
        self.sample_test_data_index = None


        # building the directories
        utils.create_directory(self.output_path)
        utils.create_directory(os.path.join(self.output_path, 'outputs'))
        utils.create_directory(self.model_path)

        # copying config file
        dest = os.path.join(self.output_path, cfg_file.split('/')[-1])
        copyfile(cfg_file, dest)





    # ============================================================================================================
    # ============================================================================================================
    def setup_model(self):
        """
        Sets up network, dataloader, optimizers
        """


        # building the dataset
        self.dataset = Datasets.Dataset(data_dir=self.data_path,
                                        sensor_location=self.location,
                                        step_label_use=self.step_label_tag,
                                        stride_label_use=self.stride_label_tag,
                                        estimate_trajectory=self.opt.estimate_trajectory,
                                        trajectory_labels_use=self.trajectory_labels,
                                        valid_size=self.valid_size,
                                        batch_size=self.batch_size,
                                        input_size=self.input_dim[-1],
                                        enable_augmentation=self.enable_augmentation,
                                        num_roll_aug=self.num_roll_augmentation,
                                        shuffle=True,
                                        seed=0)

        # self.train_loader, self.valid_loader = self.dataset.get_train_validation_set()
        train_sampler, valid_sampler = self.dataset.get_train_validation_set()

        self.train_loader = torch.utils.data.DataLoader(self.dataset,
                                                        batch_size=self.batch_size,
                                                        sampler=train_sampler)

        # building validation dataloader and dataset: different due to augmentation in training
        self.valid_dataset = Datasets.Dataset(data_dir=self.data_path,
                                              sensor_location=self.location,
                                              step_label_use=self.step_label_tag,
                                              stride_label_use=self.stride_label_tag,
                                              estimate_trajectory=self.opt.estimate_trajectory,
                                              trajectory_labels_use=self.trajectory_labels,
                                              valid_size=self.valid_size,
                                              batch_size=self.valid_batch_size,
                                              input_size=self.input_dim[-1],
                                              enable_augmentation=False,
                                              num_roll_aug=self.num_roll_augmentation,
                                              shuffle=False)

        self.valid_loader = torch.utils.data.DataLoader(self.valid_dataset,
                                                        batch_size=self.batch_size,
                                                        sampler=valid_sampler)

        # sample indexes
        self.sample_test_data_index = sample(self.valid_loader.dataset.data_list, self.num_sample_test)

        # setting up device
        torch.backends.cudnn.fastest = True
        print(torch.cuda.is_available())

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)


        # here we load the model
        self.model = models.my_model(self.input_dim, self.output_dim, self.opt, self.device)
        self.model = self.model.to(self.device)

        # loading pretrained cnn weights:
        if not self.opt.cnn_pt == 'none':
            cnn_state_dict = torch.load(self.opt.cnn_pt)['cnn_state_dict']
            self.model.CNN.load_state_dict(cnn_state_dict)

            # deactivating the grad
            if self.opt.cnn_freeze:
                print('freezing the first cnn layers')
                for lay in self.model.CNN.parameters():
                    lay.requires_grad = False


        # ********* setting up model parameters
        # optimizers
        self.CNN_optimizer = torch.optim.Adam(self.model.CNN.parameters(), lr=self.learning_rate)

        if not self.opt.net_type == 'identity':
            self.core_optimizer = torch.optim.Adam(self.model.core_network.parameters(), lr=self.learning_rate)

        self.regressor_optimizer = torch.optim.Adam(self.model.Regressor.parameters(), lr=self.learning_rate)

        # loss functions
        if self.opt.cost_function == 'MSELoss':
            self.loss_function = torch.nn.MSELoss(reduction='mean')
        elif self.opt.cost_function == 'SmoothL1Loss':
            self.loss_function = torch.nn.SmoothL1Loss(reduction='mean')
        elif self.opt.cost_function == 'L1Loss':
            self.loss_function = torch.nn.L1Loss(reduction='mean')

        if self.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.CNN_optimizer, milestones=[30, 80], gamma=0.01)


    # ======================================================================================================
    # ======================================================================================================
    def analyse_dataset(self):
        """
        Analysis of the dataset, returns the label distribution
        :return:
        """
        # analysing the train and test dataset
        utils.analyse_data_set(self.train_loader, os.path.join(self.output_path, 'train_data.png'))
        utils.analyse_data_set(self.valid_loader,  os.path.join(self.output_path, 'test_data.png'))

        utils.get_label_distribution(self.train_loader, self.valid_loader, os.path.join(self.output_path))



    # =====================================================================================================
    # =====================================================================================================
    def training_loop(self):
        """
        Computes the training and testing loop
        """
        if not self.opt.estimate_trajectory:
            self.analyse_dataset()


        # training and evaluating model
        for epoch in range(self.max_epochs):
            # train
            self.train_epoch(epoch)

            # eval
            if epoch % self.save_frequency == 0:

                # test the model
                self.test_epoch(epoch)

                # Saving the model if it is the best one
                flag = self.best_validation_loss > self.valid_loss
                if self.best_validation_loss > self.valid_loss:
                    self.best_validation_loss = self.valid_loss

                # visualizing the results achieved so far (saved in a directory)
                if self.opt.estimate_trajectory:
                    self.visualize_trajectory(epoch, flag, self.label_names)
                else:
                    self.Visualize(epoch, flag=flag)

                # saving the results into a .txt file and saving the model
                self.save_current_state(epoch=epoch, flag=flag)


        # saving trained model
        trained_model_name = 'model_trained.pwf'
        model_path = os.path.join(self.output_path, 'models')
        dir_existed = utils.create_directory(model_path)
        torch.save(self.model.state_dict(), os.path.join(model_path, trained_model_name))
        print("Training Completed!!")


    # =====================================================================================================
    # =====================================================================================================
    def train_epoch(self, epoch):
        """
        Method that computed a training epoch of the network

        Args
        ----
        epoch: Integer
            Current training epoch
        """

        #
        self.model.train()
        self.model.CNN.train()
        self.model.core_network.train()
        self.model.Regressor.train()

        # Create list to store loss value to display statistics
        loss_function_value_list = []

        self.train_output = np.zeros((0, self.output_dim[-1]))
        self.train_label = np.zeros((0, self.output_dim[-1]))

        # iterate batch by bacth over the train loader
        pbar = tqdm(total=len(self.train_loader), desc='Training epoch {}'.format(epoch), position=0, leave=True)
        for i, (sensor_data, labels) in enumerate(self.train_loader):

            # merging augmentation with the batch if augmented
            if len(sensor_data.shape) == 5:
                dim0, dim1, dim2, dim3, dim4 = sensor_data.shape
                sensor_data = sensor_data.view(-1, dim2, dim3, dim4)

                dim0, dim1, dim2, dim3 = labels.shape
                labels = labels.view(-1, dim3)

            # sending sensor data and labels to the device
            sensor_data = sensor_data.to(self.device)
            labels = labels.to(self.device)
            labels = labels.long()

            # reseting the gradients
            self.CNN_optimizer.zero_grad()

            if not self.opt.net_type == 'identity':
                self.core_optimizer.zero_grad()

            self.regressor_optimizer.zero_grad()

            # Forward pass TODO: correct te data dimension
            # sensor_data = sensor_data.squeeze()

            outputs = self.model.Regressor(self.model.core_network(self.model.CNN(sensor_data.float())))
            outputs = outputs.double()

            # Loss:  = mean(err_1**2 + err_2 **2)/2
            # todo: check dimensions
            # TODO: loss is mean(power2(diff))! no sqrt!
            loss = self.loss_function(input=outputs.squeeze(), target=labels.double().squeeze())
            loss_function_value_list.append(loss)

            # Backward and optimize
            loss.backward()

            # TODO: why filtering and requires_grad = False does not work!

            self.CNN_optimizer.step()

            if not self.opt.net_type == 'identity':
                self.core_optimizer.step()

            self.regressor_optimizer.step()

            # updating the scheduler
            if self.scheduler:
                self.scheduler.step()

            # computing accuracy on the training set
            outputs = outputs.detach().cpu()

            # updating batch error
            self.train_output = np.concatenate((self.train_output, outputs.numpy()))
            self.train_label = np.concatenate((self.train_label, labels.view((-1, self.output_dim[-1])).cpu().numpy()))

            if i % 5 == 0:
                pbar.update(5)

        # Print loss
        if epoch % self.save_frequency == 0:
            self.train_loss = utils.get_loss_stats(loss_function_value_list)
            self.train_loss = self.train_loss.item()
            self.loss_total_train.append(self.train_loss)
            self.train_mse.append(np.mean(np.sqrt(np.sum((self.train_output - self.train_label) ** 2, 1))))



        print("\n")


    # =======================================================================================
    # =======================================================================================
    def test_epoch(self, epoch):
        """
        Method that computing a validation epoch of the network

        Args
        ----
        epoch: Integer
            Current training epoch
        """

        self.model.eval()
        self.model.CNN.eval()
        self.model.core_network.eval()
        self.model.Regressor.eval()

        num_valid_data = 0
        accuracy_on_labels = 0
        validation_outputs = []
        validation_labels = []
        self.valid_output = np.zeros((0, self.output_dim[-1]))
        self.valid_label = np.zeros((0, self.output_dim[-1]))


        pbar = tqdm(total=len(self.valid_loader), desc='validation epoch {}'.format(epoch), position=0, leave=True)

        with torch.no_grad():
            loss_list = []
            for i, (sensor_data, labels) in enumerate(self.valid_loader):

                # merging augmentation with the batch if augmented
                if len(sensor_data.shape) == 5:
                    dim0, dim1, dim2, dim3, dim4 = sensor_data.shape
                    sensor_data = sensor_data.view(-1, dim2, dim3, dim4)

                    dim0, dim1, dim2, dim3 = labels.shape
                    labels = labels.view(-1, dim3)

                # sending the data to the device
                sensor_data = sensor_data.to(self.device)
                labels = labels.to(self.device)
                labels = labels.long()

                outputs = self.model.Regressor(self.model.core_network(self.model.CNN(sensor_data.float())))
                outputs = outputs.double()

                loss_list.append(self.loss_function(input=outputs.squeeze(), target=labels.squeeze().double()))

                # computing accuracy on the test set
                outputs = outputs.detach().cpu()

                # saving the output values
                self.valid_output = np.concatenate((self.valid_output, outputs.numpy()))


                self.valid_label = np.concatenate((self.valid_label, labels.view((-1, self.output_dim[-1])).cpu().numpy()))
                # predicted_labels = label_list[np.argmax(outputs, axis=1)]
                batch_acc = np.abs(outputs.squeeze() - labels.squeeze().cpu()) <= (self.acc_bound_rel * (labels.cpu()))
                accuracy_on_labels += batch_acc.sum()
                num_valid_data += len(sensor_data)

                pbar.update(1)


        # validation set statistics
        self.valid_loss = utils.get_loss_stats(loss_list)
        self.valid_loss = self.valid_loss.item()
        self.validation_loss_total.append(self.valid_loss)

        if not self.opt.estimate_trajectory:
            self.valid_mae_each = np.concatenate((
                self.valid_mae_each,
                (np.abs(self.valid_output - self.valid_label)).mean(axis=0)[np.newaxis, :]
            ))
            self.valid_std_each = np.concatenate((
                self.valid_std_each,
                np.std(self.valid_output - self.valid_label, axis=0)[np.newaxis, :]
            ))

        else:
            self.valid_mae_each = np.concatenate((
                self.valid_mae_each,
                (np.abs(self.valid_output - self.valid_label)).mean()[np.newaxis, np.newaxis]
            ))

            self.valid_std_each = np.concatenate((
                self.valid_std_each,
                np.mean(np.std(self.valid_output - self.valid_label, axis=0))[np.newaxis, np.newaxis]
            ))

        tot_eval_mse = np.mean(np.sqrt(np.sum((self.valid_output - self.valid_label) ** 2, 1)))

        # tot_eval_mse = np.sqrt(np.mean((self.valid_output - self.valid_label) ** 2))
        tot_eval_std = np.std(np.sqrt(np.mean((self.valid_output - self.valid_label)**2, axis=1)))
        # self.eval_error.append(np.mean([np.sqrt(l.item()) for l in loss_list]))
        self.all_eval_mse_error.append(tot_eval_mse)
        # self.eval_variance.append(np.std([np.sqrt(l.item()) for l in loss_list]))
        self.all_eval_variances.append(tot_eval_std)

        print('validation error: {}'.format(tot_eval_mse))
        print("\n")



    # =============================================================================================
    # =============================================================================================
    def calculate_and_save_error_terms(self):
        pass


    # =====================================================================================================
    # =====================================================================================================
    def visualize_trajectory(self, epoch, flag, label_names):
        """
        Visualization for trajectory estimation task

        :param epoch:
        :param flag:
        :param label_names:
        :return:
        """


        # **** sample input and output for trajectory values
        sample_outputs = []
        sample_labels = []

        # calculating the outputs of samples
        for s, t, i in self.sample_test_data_index:
            test_sample, sample_label = self.valid_loader.dataset.__getitem__([s, t, i])
            test_sample = torch.Tensor(test_sample[np.newaxis, :, :, :]).to(self.device)
            sample_output = self.model.Regressor(self.model.core_network(self.model.CNN(test_sample.float())))

            # appending the info
            sample_outputs.append(sample_output.detach().cpu().tolist())
            sample_labels.append(sample_label)

        # plotting one random sample
        plt.figure()
        plt.plot(test_sample.squeeze().transpose(1, 0).cpu().numpy())
        plt.grid()
        plt.title('one random sample')
        plt.savefig(os.path.join(self.output_path, 'outputs', 'random_sample.png'))

        plt.close()


        # plotting some sample input and output predicted values for each
        gt.trajectory_vs_predicted(sample_outputs, sample_labels, self.label_names,
                                   os.path.join(self.output_path, 'outputs', 'predictions_epoch_{}.png'.format(epoch))
                                   )

        # plotting the error and variance plot
        gt.error_variance_history_bar_plot(self.valid_mae_each,
                                           self.valid_std_each,
                                           epoch,
                                           os.path.join(self.output_path, "Error_and_std_plot.png")
                                           )

        # **** plotting loss plot
        gt.plot_loss(self.loss_total_train, self.validation_loss_total,
                     self.save_frequency,
                     os.path.join(self.output_path, 'loss.png'))

        # ******* Visualize the kernels
        gt.visualize_cnn_kernels(self.model.CNN.conv,
                                 self.output_path
                                 )

        # ******** bland-altman plot
        if flag:

            gt.bland_altman_plot(np.reshape(self.valid_output, (-1, 1)),
                                 np.reshape(self.valid_label, (-1, 1)),
                                 self.label_names, epoch,
                                 os.path.join(self.output_path, 'Bland-Altman.png')
                                 )


    # =============================================================================================
    # =============================================================================================
    def Visualize(self, epoch, flag):
        """
        This method saves plots to visualize the progress achieved so far.
        :param epoch:
        :param validation_labels:
        :param validation_outputs:
        :param num_sample_data:
        :return:
        """


        # **** sample input and output for all predicted labels
        sample_outputs = []
        sample_labels = []

        # calculating the outputs of samples
        for s, t, i in self.sample_test_data_index:
            test_sample, sample_label = self.valid_loader.dataset.__getitem__([s, t, i])
            test_sample = torch.Tensor(test_sample[np.newaxis, :, :, :]).to(self.device)
            sample_output = self.model.Regressor(self.model.core_network(self.model.CNN(test_sample.float())))

            # appending the info
            sample_outputs.append(sample_output.detach().cpu().tolist())
            sample_labels.append(sample_label)

        # plotting one random sample
        plt.figure()
        plt.plot(test_sample.squeeze().transpose(1, 0).cpu().numpy())
        plt.grid()
        plt.title('one random sample')
        plt.savefig(os.path.join(self.output_path, 'outputs', 'random_sample.png'))
        plt.close()


        # *** predicted vs true value
        gt.plot_labels_vs_predicted_class(sample_outputs, sample_labels,
                                          self.label_names,
                                          os.path.join(self.output_path, 'outputs', 'predictions_epoch_{}.png'.format(epoch)))

        # ***** error and variance plot of each of the target labels individually
        gt.error_variance_history_bar_plot(self.valid_mae_each,
                                           self.valid_std_each,
                                           epoch,
                                           os.path.join(self.output_path, "Error_and_std_plot.png")
                                           )

        # ** plotting loss plot
        gt.plot_loss(self.loss_total_train, self.validation_loss_total,
                     self.save_frequency,
                     os.path.join(self.output_path, 'loss.png'))


        # ****** Visualize the kernels
        gt.visualize_cnn_kernels(self.model.CNN.conv,
                                 self.output_path
                                 )


        # ******** bland-altman plot
        if flag:
            gt.bland_altman_plot(self.valid_output, self.valid_label,
                                 self.label_names, epoch,
                                 os.path.join(self.output_path, 'Bland-Altman.png')
                                 )



    # ==========================================================================================
    # ==========================================================================================
    def save_current_state(self, epoch, flag):
        """
        This method is used to save the current results as a .txt file
        :return:
        """

        # adding experiment explanation:
        exp_file_name = os.path.join(self.output_path, 'res.txt')
        info = 'Epoch:{}, mse_tr={:.5f}, mse_te={:.5f}, std_te={:.5f}, '.format(
            epoch, self.loss_total_train[-1], self.all_eval_mse_error[-1], self.all_eval_variances[-1])

        info_each_data = ''
        for idx, lbl in enumerate(self.step_label_tag + self.stride_label_tag):
            info_each_data += 'mae_' + lbl + '={:.5f}, '.format(self.valid_mae_each[-1, idx])
        info_each_data += '\n'

        with open(exp_file_name, 'a') as f:
            f.write(info + info_each_data)

        if flag:
            # Saving the best model every save_frequency epochs
            if epoch % self.save_frequency == 0:
                trained_model_name = 'model_epoch_' + str(epoch+1) + '.pwf'
                torch.save(self.model.state_dict(), os.path.join(self.model_path, trained_model_name))



if __name__ == "__main__":

    os.system("clear")
    cfg_file_base = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cfg')
    cfg_file_names = ['I.cfg']  #['II.cfg', 'III.cfg', 'IIII.cfg']

    for cfg_file_name in cfg_file_names:
        print(os.path.join(cfg_file_base, cfg_file_name))
        trainer = Trainer(os.path.join(cfg_file_base, cfg_file_name))
        trainer.setup_model()
        trainer.training_loop()

