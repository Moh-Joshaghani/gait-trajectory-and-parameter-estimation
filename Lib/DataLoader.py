"""
This class loades the data!
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d

# libraries to read data
data_path = '.\data\Vicon_running_data_set'
import sys
sys.path.insert(0, data_path)
from data.Vicon_running_data_set.vicon_running_data_set import vicon_running_data_set




# ========================================================================
#               CLASSES
class Dataset(Dataset):
    def __init__(self, data_dir='', foot='left', interp=False, sensor_location='Cavity',
                 target_label_use=['orientation_x'],
                 step_label_use=['Sole_Angle', 'Impact_Angle', 'Max_Pro_Angle', 'Initial_Supination'],
                 stride_label_use=['Sole_Angle', 'Impact_Angle', 'Max_Pro_Angle', 'Initial_Supination'],
                 trajectory_labels_use=None,
                 estimate_trajectory=True,
                 train_size=0.8, valid_size=0.2, shuffle=False,
                 input_size=300, batch_size=128, seed=13,
                 enable_augmentation=False, num_roll_aug=1):

        super().__init__()
        self.shape = None
        self.data_dir = data_dir
        self.train_size = train_size
        self.valid_size = valid_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.random_seed = seed
        self.data_dict = None
        self.data_list = []
        self.input_size = input_size

        # Info about splitted data:
        self.train_examples = None
        self.test_examples = None
        self.estimate_trajectory = estimate_trajectory

        # features to use
        self.target_label_use = target_label_use
        self.interp = interp
        self.step_label_use = step_label_use
        self.stride_label_use = stride_label_use
        self.trajectory_labels_use = trajectory_labels_use

        # ‌assert (bool(len(step_label_use) == 0) != bool(len(stride_label_use) == 0)),\
        #    'Only one of the stride labels and step labels should have values'

        # augmentation
        self.enable_augmentation = enable_augmentation
        self.num_roll = num_roll_aug
        self.augment_function = Roll_data(self.num_roll)

        # Instantiating the main dataset object
        self.data_set = vicon_running_data_set(data_dir, sensor_location, foot)

        # data information
        self.foot = foot
        self.sensor_location = sensor_location

        # constants
        self.g = 9.8
        self.range_acc_sensor = 9.8 * 12
        self.range_gyro_sensor = 2000.0

        # Loading the available subjects and trials
        self.__load_valid_dir()

    # =====================================================================
    def __load_valid_dir(self):
        """
        loads all labels and directories!
        :return:
        """
        self.data_set = vicon_running_data_set(self.data_dir)
        self.data_dict = self.__get_valid_subject_trials()

        # converting the dict to a list
        for key, values in self.data_dict.items():
            temp = [[key, value, rr] for value, rr in values]
            self.data_list.extend(temp)

        return


    # ====================================================================================
    def __len__(self):
        return self.shape


    # =====================================================================================
    def __getitem__(self, idx):
        """

        :param idx: the triplet of 'subject, trial, index'
        :return:
        """


        if self.estimate_trajectory:
            sensor_data, labels = self.__get_data_and_trajectory_values(idx)
        else:
            sensor_data, labels = self.__get_data_and_classification_labels(idx)

        return np.asarray(sensor_data), np.asarray(labels)


    # ================================================================================================
    # ================================================================================================
    def __get_valid_subject_trials(self):


            """
            This function returns the valid strides and subjects, which later is used to load
            decent data
            :param foot:
            :param location:
            :param stride_segmentation:
            :return:
            """

            # get list of all subjects
            subjects = self.data_set.get_available_subjects()

            # the subject and trials of the valid data
            valid_data_list = {}

            # iterating over all subject and investigate the data
            for subj in tqdm(subjects, desc='Reading the valid data lists'):

                # get list of all trials
                trials = self.data_set.get_trials(subj)

                for trial in trials:
                    num_strides, strides_HS = \
                        self.data_set.extract_strides_from_trial(subj, trial, self.foot,
                                                                 self.sensor_location,
                                                                 stride_segmentation='HS')

                    a = self.data_set.get_goldstandard_step(subj, trial, self.foot)
                    b = self.data_set.get_goldstandard_stride(subj, trial, self.foot)

                    if (num_strides > 0) and (len(strides_HS) > 0):
                        if subj in valid_data_list.keys():

                            [valid_data_list[subj].append([trial, idx]) for idx in range(len(strides_HS))]

                        else:
                            valid_data_list.update({subj: [[trial, 0]]})
                            [valid_data_list[subj].append([trial, idx + 1]) for idx in range(len(strides_HS) - 1)]

            return valid_data_list


    # =====================================================================================
    # =====================================================================================
    def __get_data_and_classification_labels(self, idx, typ='float64'):
        """
        reads the sensor data and label of the given index
        :return:
        """
        if not type(idx) == list:
            idx = [idx]


        if not type(idx) == list:
            idx = [idx]

        # selected trial and object and data from index:
        # current_list = [self.data_list[index] for index in idx]
        current_list = [idx]

        # getting a list of a list of panda dataframes!
        sensor_data_lst = []
        labels_list = []

        # reading labels and sensor data from the list of available trials and objects
        for s, t, i in current_list:

            # *** Processing the sensor values
            # reading the values
            n, dummy_data = self.data_set.extract_strides_from_trial(s, t, self.foot, self.sensor_location,
                                                                     stride_segmentation='HS')

            # picking the experiment's index
            dummy_data = dummy_data[i]

            # Normalizing the sensor data range
            dummy_data = self.normalize_max(dummy_data, ['accX', 'accY', 'accZ'], self.range_acc_sensor)
            dummy_data = self.normalize_max(dummy_data, ['gyroX', 'gyroY', 'gyroZ'], self.range_gyro_sensor)

            # zero padding
            l = len(dummy_data)
            dummy_data = dummy_data.append(pd.DataFrame((self.input_size-l)*[len(dummy_data.columns) * [0]],
                                           columns=dummy_data.columns),
                                           ignore_index=True)

            # Aggregating all the data, and correcting the dimensions
            sensor_data_lst.append(np.transpose(dummy_data.values.tolist()).tolist())

            # *** Processing labels

            # reading step labels
            if not len(self.step_label_use) == 0:
                gold_standard_step_labels = self.data_set.get_goldstandard_step(s, t, self.foot)
                gold_standard_step_labels = gold_standard_step_labels[self.step_label_use]
                gold_standard_step_labels = gold_standard_step_labels.loc[i+1].values.tolist()
            else:
                gold_standard_step_labels = []

            # reading stride labels
            if not len(self.stride_label_use) == 0:
                gold_standard_stride_labels = self.data_set.get_goldstandard_stride(s, t, self.foot)
                gold_standard_stride_labels = gold_standard_stride_labels[self.stride_label_use]

                gold_standard_stride_labels = gold_standard_stride_labels.loc[i].values.tolist()

            else:
                gold_standard_stride_labels = []

            labls = gold_standard_step_labels + list(np.array(gold_standard_stride_labels)*100)
            # Aggregating all the labels as list
            labels_list.append(labls)

        # the size are:
        # sensor_data_lst: B*C*D
        # labels_list: B*D

        sensor_data_lst = np.asarray(sensor_data_lst).swapaxes(0, 1).tolist()

        # Augmentation
        if self.enable_augmentation:
            sensor_data_lst, labels_list = self.augment_function(sensor_data_lst, labels_list)

        # changing the axis to become Channel*1*signal_length
        return sensor_data_lst, labels_list


    # =====================================================================================
    # =====================================================================================
    def __get_data_and_trajectory_values(self, idx, typ='float64'):
        """
        gets the sensor data and related trajectory values
        :return:
        """
        if not type(idx) == list:
            idx = [idx]



        # selected trial and object and data from index:
        # current_list = [self.data_list[index] for index in idx]
        current_list = [idx]

        # getting a list of a list of panda dataframes!
        sensor_data_lst = []
        trajectory_list = []

        # reading labels and sensor data from the list of available trials and objects
        for s, t, i in current_list:

            # *** Processing the sensor values
            # reading the values
            n, dummy_data = self.data_set.extract_strides_from_trial(s, t, self.foot, self.sensor_location,
                                                                     stride_segmentation='HS')

            # picking the experiment's index
            dummy_data = dummy_data[i]
            # Normalizing the sensor data range
            dummy_data = self.normalize_max(dummy_data, ['accX', 'accY', 'accZ'], self.range_acc_sensor)
            dummy_data = self.normalize_max(dummy_data, ['gyroX', 'gyroY', 'gyroZ'], self.range_gyro_sensor)

            # zero padding
            sensor_data_length = len(dummy_data)
            dummy_data = dummy_data.append(pd.DataFrame((self.input_size-sensor_data_length)*[len(dummy_data.columns) * [0]],
                                           columns=dummy_data.columns),
                                           ignore_index=True)

            # Aggregating all the data, and correcting the dimensions
            sensor_data_lst.append(np.transpose(dummy_data.values.tolist()).tolist())

            # ________________________________
            # *** Processing trajectory values

            # reading step trajectory values
            n, dummy_trajectory = self.data_set.extract_strides_vicon(s, t, self.foot, 'HS')
            # picking the experiment's index
            dummy_trajectory = dummy_trajectory[i]
            # selecting indexes
            trajectory = dummy_trajectory[self.trajectory_labels_use]
            # interpolating to have same length input and output
            trajectory = self.interpolate_sensor_data(sensor_data_length, trajectory)
            # zero-centring the trajectory
            trajectory = self.zero_center(trajectory, trajectory.columns)

            # same padding (because the sensor data is zero added means that the target value should not change
            trajectory = trajectory.append(pd.DataFrame((self.input_size-sensor_data_length)*[trajectory.values.tolist()[-1]],
                                           columns=trajectory.columns),
                                           ignore_index=True)

            # Aggregating all the labels as list
            trajectory_list.append(np.transpose(trajectory.values.tolist()).tolist())


        # the size are:
        #   sensor_data_lst: B*C*1*D
        #   trajectory_list: B*D
        sensor_data_lst = np.asarray(sensor_data_lst).swapaxes(0, 1).tolist()


        return sensor_data_lst, trajectory_list


    # ===================================================================================
    # ===================================================================================
    def absolute_to_releative(self, in_datas, cols):
        """
        Converts the path from absolute to releative
        :param in_datas: list of data
        :param cols: parameters to consider
        :return:
        """

        for idx, in_data in enumerate(in_datas):
            for col in cols:

                col_data = in_data[col].tolist()

                col_data_l = list(col_data)
                col_data_l.append(col_data[-1])
                col_data_l = np.array(col_data_l)

                col_data_r = list(col_data)
                col_data_r.insert(0, 0)
                col_data_r = np.array(col_data_r)

                col_data_rel = list(col_data_l - col_data_r)

                # replacing the column
                in_data[col] = col_data_rel[:-1]

            # replacing the pd frame
            in_datas[idx] = in_data

        return in_datas


    # ===================================================================================
    # ===================================================================================
    def get_desired_labels(self, in_labels):
        """
        extracts desired labels from the input
        :param in_labels: the vicon dataset labels datframe
        :return: the overall displacement
        """

        # passing in label as a list
        if not type(in_labels) == list:
            in_labels = [in_labels]

        out_labels = []
        # out_label_columns = ['disp x', 'disp y', 'disp z', 'disp total']
        out_label_columns = ['disp total']

        # calculating the disired parameters for each label
        for in_label in in_labels:

            # zero centring the data and converting to centimeter
            in_label = self.zero_center(in_label, ['loc_x', 'loc_y', 'loc_z'])
            loc_x_all = list(np.asarray(in_label['loc_x'].tolist()) / 1000.0)  # /max(abs(np.asarray(in_label['loc_x'].tolist())))
            loc_y_all = (in_label['loc_y'] / 1000.0).tolist()  # /max(abs(np.asarray(in_label['loc_y'].tolist())))
            loc_z_all = (in_label['loc_z'] / 1000.0).tolist()   # /max(abs(np.asarray(in_label['loc_z'].tolist())))

            displacement_x = loc_x_all[-1] - loc_x_all[0]
            displacement_y = loc_y_all[-1] - loc_y_all[0]
            displacement_z = loc_z_all[-1] - loc_z_all[0]

            # calculating the total displacement
            stride_length = np.sqrt(displacement_x**2 + displacement_y**2 + displacement_z**2)

            # out_labels.append([displacement_x, displacement_y, displacement_z, displacement_total])
            out_labels.append([stride_length])


        # forming the pandas dataframe
        out_labels_pd = pd.DataFrame(out_labels, columns=out_label_columns)

        return out_labels, out_labels_pd


    # ==================================================================================
    # ==================================================================================
    def normalize_max(self, in_data, cols, norm_factor):
        """

        :param in_data: pandas dataframe, including cols columns to be normalized
        :param cols: columns to be normalized
        :param norm_factor: factor with which the columns will be normalized
        :return: normalized data
        """

        for col in cols:
            # Normalizing the column
            col_data = in_data[col]/norm_factor

            # replacing the column
            in_data[col] = col_data


        return in_data

    # ==================================================================================
    # ==================================================================================
    def zero_center(self, in_data, cols):
        """

        :param in_data:
        :param cols:
        :return:
        """


        for col in cols:

            col_data = in_data[col]
            col_data_centered = list(np.asarray(col_data.tolist()) - col_data.tolist()[0])

            # replacing the column
            in_data[col] = col_data_centered




        return in_data


    # ===================================================================================
    # ===================================================================================
    def interpolate_sensor_data(self, target_length, trajectory):
        """
        This method interpolates labels to have the same size as sensor data.
        :param trajectory: pandas
        :param sensors_data: pandas
        :return:
        """


        trajectory_interpolated = pd.DataFrame(columns=trajectory.columns)

        # old x
        x = np.linspace(0, np.shape(trajectory)[0]-1, np.shape(trajectory)[0])

        # getting the new x
        x_new = np.linspace(0, trajectory.shape[0] - 1,target_length)

        for col in trajectory.columns:

            col_data = trajectory[col].tolist()
            f = interp1d(x, col_data)
            col_data_interpolated = f(x_new)
            trajectory_interpolated[col] = col_data_interpolated


        return trajectory_interpolated

    # ===================================================================================
    # ===================================================================================
    def no_augmentation(self):
        """
        sets augmentation to False, for validation set
        :return:
        """
        self.enable_augmentation = False

    def set_augmentation(self):
        """
        sets augmentation to False, for validation set
        :return:
        """
        self.enable_augmentation = True

    # ====================================================================================
    # ====================================================================================
    def get_list_of_pandas(self, in_pandas_, cols_to_use, apply_size_limit=True, typ='float64'):
        """
        converts a list of input pandas to a list and zero pads
        :param in_pandas_:
        :param typ:
        :param apply_size_limit:
        :param cols_to_use:
        :return:
        """

        if not type(in_pandas_) == list:
            in_pandas = [in_pandas_]
        else:
            in_pandas = in_pandas_

        # number of channels
        num_channels = len(in_pandas[0].columns)

        # initializing the data: B*C*self.input_size
        all_data = np.zeros((len(in_pandas), num_channels, self.input_size))

        # listifying every panda dataframe in the list
        for idx, in_panda in enumerate(in_pandas):
            dummy_data_ = [in_panda[c].tolist() for c in in_panda.columns]

            # applying input size limitation
            if apply_size_limit:
                len_sensor_data = np.shape(dummy_data_)[1]
                if len_sensor_data < self.input_size:
                    dummy_data = np.asarray([np.asarray(dummy_col + (self.input_size - len_sensor_data)*[0], dtype=typ)
                                  for dummy_col in dummy_data_])
                else:
                    dummy_data = np.asarray([np.asarray(dummy_col[:self.input_size], dtype=typ) for dummy_col in dummy_data_])
            else:
                dummy_data = dummy_data_

            all_data[idx] = dummy_data

        return all_data


    # =====================================================================================
    # =====================================================================================
    def get_train_validation_set(self):
        """
        Creates the train/validation split and returns the data loaders, splits based
        on the subjects
        """

        # counting the number of the data
        num_subjects_data = [l[0] for l in self.data_list]
        num_subjects = len(set(num_subjects_data))
        split = int(np.floor(self.valid_size * num_subjects))

        num_train = len(self.data_list)
        subj_indices = list(set(num_subjects_data))

        # randomizing train and validation set
        if(self.shuffle):
            np.random.seed(self.random_seed)
            np.random.shuffle(subj_indices)

        # getting idx for train and validation
        train_subj, valid_subj = subj_indices[split:], subj_indices[:split]
        train_idx = [l for l in self.data_list if l[0] in train_subj]
        valid_idx = [l for l in self.data_list if l[0] in valid_subj]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)


        self.train_examples = len(train_idx)
        self.valid_examples = len(valid_idx)


        print("\n")
        print(f"Total number of examples is {num_train}")
        print(f"Size of training set is approx {self.train_examples}")
        print(f"Size of validation set is approx {self.valid_examples}")

        # return train_loader, valid_loader
        return train_sampler, valid_sampler



    # ========================================================================
    # ========================================================================
    def get_test_set(self):
        pass


# =====================================================================================
# =====================================================================================
class Roll_data(object):
    """
    randomly shifts left or right the input

    Args:
         :parameter num_roll: Number of times the data is rolled

    """

    def __init__(self, num_roll):
        self.num_roll = num_roll

    # ===============================
    def __call__(self, sample, label):
        """
        performs the rolling!
        :param sample: input, np array : 1 * c * n
        :param label: list [[]]: 1 * 1
        :return:
        """
        new_sample = np.zeros((np.shape(sample)[0]*self.num_roll,
                               np.shape(sample)[1],
                               np.shape(sample)[2]))

        new_sample = np.zeros((self.num_roll,
                               np.shape(sample)[0],
                               np.shape(sample)[1],
                               np.shape(sample)[2]))

        new_labels = []
        # calculating roll amounts:
        roll_range = np.shape(sample)[-1]
        roll_step = int(roll_range/self.num_roll)
        roll_array = np.linspace(0, roll_range - roll_step, self.num_roll, dtype=int)
        roll_array = np.random.randint(0, roll_range, self.num_roll)


        # rolling
        for r_idx in range(self.num_roll):
            new_sample[r_idx,:, :, :] = np.roll(sample,
                                                roll_array[r_idx],
                                                axis=1)

        new_labels.extend(self.num_roll * [label])



        return new_sample, new_labels


# ---------------------------------------------------------------------------------------
""""
if __name__ == '__main__':

    data_path = '/mnt/5476e3a8-ff8d-4296-8bd3-5d8f813d0507/ghost/Archive/Courses/MLTS/Project/trajectory-estimation/data/Vicon_running_data_set'
    ds = Dataset(data_dir=data_path)
    a, b = ds.__getitem__([10, 20])
    ds.get_train_validation_set()

    a = 1

"""

