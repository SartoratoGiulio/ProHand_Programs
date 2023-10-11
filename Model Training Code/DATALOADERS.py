# import scipy
import numpy as np
import os
# import pathlib
import re
# import pandas as pd
# import matplotlib.pyplot as plt
import torch
# from sklearn import metrics
from torch.utils import data
import torch.functional as F
from math import log2
# from scipy.signal import butter, filtfilt, freqz, iirnotch
from Data_Processing_Utils import windowing_signal, mapping_labels


# %% Personalized dataloader to work with dataframes in order to extract a sample (400 raws) based on
# personalized index & groupby function
class Pandas_Dataset(data.Dataset):

    def __init__(self, df_grouped_by_samples, return_sub=False):
        self.grouped = df_grouped_by_samples
        self.channels = [i for i in df_grouped_by_samples.obj.columns if 'ch' in i]
        self.indices = list(df_grouped_by_samples.indices)
        self.return_sub = return_sub

    def __len__(self):
        return len(self.grouped)

    def __getitem__(self, index):
        picked_smp = self.grouped.get_group(self.indices[index])
        # Picking only the channels columns from a single sample
        sample = torch.tensor(picked_smp.loc[:, self.channels].values).type(torch.float32)
        # picking only one label for each sample
        label = torch.tensor(picked_smp.loc[:, ['label']].head(1).values[0][0]).type(torch.int8)
        angle = torch.tensor(picked_smp.loc[:, ['angle']].head(1).values[0][0]).type(torch.int16)
        if self.return_sub:
            # picking only one subject
            sub = torch.tensor(picked_smp.loc[:, ['sub']].head(1).values[0][0]).type(torch.int8)
            return sample, label, sub 
        else:

            # It's missing the part in which I Normalize the data for each subject, or for each subject and channels
            # It's not a good idea to do that on the dataloader while producing results
            return sample, label


# %% General grouped dataset for dataloader
class GroupbyDataset(data.Dataset):
    def __init__(self, dataframe, groupby_col, target_col, sample_frac=1.0):
        self.dataframe = dataframe
        self.groupby_col = groupby_col
        self.target_col = target_col
        self.sample_frac = sample_frac
        self.channels = [i for i in dataframe.columns if 'ch' in i]

        self.groups = dataframe.groupby(groupby_col)
        self.indices = list(self.groups.indices)
        # self.indices = {i: group.sample(frac=sample_frac).index for i, group in self.groups}
        # self.keys = list(self.indices.keys())

    def __getitem__(self, idx):
        # group_key = self.keys[idx]
        # group_indices = self.indices[group_key]
        pick_sample = self.groups.get_group(self.indices[idx])
        sample = torch.tensor(pick_sample.loc[:, self.channels].values).type(torch.float32)
        label = torch.tensor(pick_sample.loc[:, 'label'].head(1).values[0]).type(torch.int8)
        sub = pick_sample['sub'].head(1).values[0]

        return sample, label

    def __len__(self):
        return len(self.indices)


# %% dataset for dataframe which retrieve a positive and negative sample
class dataframe_dataset_triplet(data.Dataset):

    def __init__(self, dataframe, groupby_col, target_col, sample_frac=1.0):
        self.df = dataframe
        self.groupby_col = groupby_col
        self.target_col = target_col
        self.sample_frac = sample_frac
        self.channels = [i for i in dataframe.columns if 'ch' in i]

        self.groups = dataframe.groupby(groupby_col)
        self.indices = {i: group.sample(frac=sample_frac).index for i, group in self.groups}
        self.keys = list(self.indices.keys())

        # to delete df.query
        self.labels = np.unique(dataframe['label'])
        self.subjects = np.unique(dataframe[target_col])

        self.sample_indices_by_label_sub = {}
        for label, sub in dataframe[['label', 'sub']].drop_duplicates().values:
            self.sample_indices_by_label_sub[(label, sub)] = dataframe.query(f"label == {label} and {'sub'} == {sub}")[
                'sample_index'].values

    def __getitem__(self, idx):
        group_key = self.keys[idx]
        group_indices = self.indices[group_key]
        sample = torch.tensor(self.df.loc[group_indices, self.channels].values).type(torch.float32)
        label = torch.tensor(self.df.loc[group_indices, 'label'].head(1).values[0]).type(torch.int8)
        sub = self.df.loc[group_indices, self.target_col].head(1).values[0]

        # # take a unique random value from groupby_col (in this case, sample index) after filtered dataframe
        # idx_sample = np.random.choice(self.df.query(f'{self.target_col} != {sub} and label == {label}')[self.groupby_col], size=1)[0]
        # # extract the casual positive sample
        # pos_sample = torch.tensor(self.df.loc[self.df[self.groupby_col] == idx_sample, self.channels].values)\
        #     .type(torch.float32)
        #
        # idx_sample = np.random.choice(self.df.query(f'label != {label}')[self.groupby_col], size=1)[0]
        # neg_sample = torch.tensor(self.df.loc[self.df[self.groupby_col] == idx_sample, self.channels].values)\
        #     .type(torch.float32)

        # After deleting query
        # positive
        # pos_sub = np.random.choice([k for k in self.subjects if k != sub])   # Changed bc in case of one single sub raises error (e.g. test_dataloader)
        pos_sub = np.random.choice(self.subjects)
        pos_indexes = self.sample_indices_by_label_sub[(int(label), pos_sub)]
        pos_sample = torch.tensor(self.groups.get_group(np.random.choice(pos_indexes, size=1)[0])[self.channels].values) \
            .type(torch.float32)
        # negative
        neg_lbl = np.random.choice([k for k in self.labels if k != label])
        neg_indexes = self.sample_indices_by_label_sub[(neg_lbl, np.random.choice(self.subjects))]
        neg_sample = torch.tensor(self.groups.get_group(np.random.choice(neg_indexes, size=1)[0])[self.channels].values) \
            .type(torch.float32)

        return sample, label, pos_sample, neg_sample

    def __len__(self):
        return len(self.keys)


# %%Split in train val and test based on repertitions (to check)
def Train_Test_Split_Repetition(Unpacked_dataset_path, exe_list_to_keep=[], sub_to_discard=[], path_to_save=None):
    """

        :param Unpacked_dataset_path: Data are in the form SubXeYrepZ.npy where it is stored the EMG signals of each channel
        :param exe_list_to_keep:      List of exercise to keep for Hand Gesture Classificaton
        :param sub_to_discard:        Choose if there are subjects to not consider (e.g. left handed, amputee)
        :param path_to_save:          Path to save dataframe files, if none is passed, will be  "DBX/Numpy"
                                        of Unpacked_dataset_path
        :return:                      Saved Numpy array already windowed
        """

    os.chdir(Unpacked_dataset_path)
    EMG_data = sorted(os.listdir())

    if path_to_save is None:
        path_to_save = os.path.join(os.path.dirname(Unpacked_dataset_path)) + '/Numpy/'
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    # Creating Train-Val-Test manually
    Sets = {'train_set': list(), 'val_set': list(), 'test_set': list()}
    counter = 0
    # -> sub 14 & 16 removed
    for name in EMG_data:
        if 'Sub' in name and int(re.findall(r'\d+', name)[0]) not in sub_to_discard:
            exe_num = int(re.findall(r'\d+', name)[1])
            if exe_num in exe_list_to_keep:
                rep = int(re.findall(r'\d+', name)[2])
                counter = counter + 1

                if (rep == 1) or (rep == 3) or (rep == 5) or (rep == 7) or (rep == 9):
                    Sets['train_set'].append(name)  # -> 1375
                elif (rep == 2) or (rep == 10):
                    Sets['val_set'].append(name)  # -> 550
                # elif (rep == 4) or (rep == 6) or (rep == 8):
                else:
                    Sets['test_set'].append(name)  # -> 825
    print('Tot_number_of_data', counter)  # -> 2750

    # Concatenating Windows & Create Numpy array
    for set_ in Sets:
        n_data = len(Sets[set_])
        print(n_data)  # -> [1375, 550,825] = [train, val, test]
        counter = 1
        windows, label = [], []
        for name in Sets[set_]:
            exe_num = int(re.findall(r'\d+', name)[1])

            print('DATA: ', counter, ' / ', n_data)
            sample = np.load(name)
            counter = counter + 1

            windows_sample = windowing_signal(sample, 20, 0.75, drop_last=False).astype(np.float32)
            label_vec = np.full(shape=windows_sample.shape[0], fill_value=mapping_labels(exe_num, 1))

            windows.append(windows_sample)
            label.append(label_vec)

        windows = np.vstack(windows)
        label = np.concatenate(label, axis=0)
        print('#' * 100)
        print(windows.shape)
        print(label.shape)
        print('#' * 100)
        if not os.path.exists(path_to_save + 'Numpy/' + set_):
            os.makedirs(path_to_save + 'Numpy/' + set_)
        np.save(path_to_save + 'Numpy/' + set_ + '/window_no_rest.npy', windows)

        from torch.utils import data
        import torch


# %% Numpy hard-code
class Dataset_window(data.Dataset):

    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = torch.tensor(self.samples[index])
        label = torch.tensor(self.labels[index])

        return sample, label


# %% For sub Classification Dataset (model for classify a subject based on samples from same exercise, a way to prove the high variability)
class Sub_Classify_Dataset(data.Dataset):

    def __init__(self, df_grouped_by_samples):
        self.grouped = df_grouped_by_samples
        self.channels = [i for i in df_grouped_by_samples.obj.columns if 'ch' in i]
        self.indices = list(df_grouped_by_samples.indices)

    def __len__(self):
        return len(self.grouped)

    def __getitem__(self, index):
        picked_smp = self.grouped.get_group(self.indices[index])
        # Picking only the channels columns from a single sample
        sample = torch.tensor(np.array(picked_smp.loc[:, self.channels])).type(torch.float32)
        # picking only one label for each sample
        label = torch.tensor(picked_smp.loc[:, ['sub']].head(1).values[0][0]).type(torch.int8)

        # picking only one subject
        # sub = torch.tensor(picked_smp.loc[:, ['sub']].head(1).values[0][0]).type(torch.int8)

        # It's missing the part in which i Normalize the data for each subject, or for each subject and channels
        # It's not a good idea to do that on the dataloader
        return sample, label


# %% KL and JS divergence
def calculate_jensen_shannon_divergence(model1, model2):
    # Get the weights of both models
    weights1 = torch.cat([param.view(-1) for param in model1.parameters()])
    weights2 = torch.cat([param.view(-1) for param in model2.parameters()])

    # Normalize the weights
    weights1 = F.softmax(weights1, dim=0)
    weights2 = F.softmax(weights2, dim=0)

    # Calculate Jensen-Shannon Divergence
    m = 0.5 * (weights1 + weights2)
    jsd = 0.5 * (F.kl_div(weights1, m) + F.kl_div(weights2, m))

    return jsd
