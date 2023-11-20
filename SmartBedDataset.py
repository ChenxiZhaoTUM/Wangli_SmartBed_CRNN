import numpy as np
import torch
from torch.utils.data import Dataset
import unittest
import os


def TensorLoader(data, isTest=False):
    if isTest:
        folder_path = data.dataDirTest
        files = [file for file in os.listdir(folder_path) if "test_inputs" in file]
        data.all_inputs = []
        data.all_targets = []

        for file in files:
            test_inputs = torch.load(os.path.join(folder_path, file))
            test_targets = torch.load(os.path.join(folder_path, file.replace('test_inputs', 'test_targets')))
            data.all_inputs.append(test_inputs)
            data.all_targets.append(test_targets)

        data.all_inputs = torch.cat(data.all_inputs, dim=0)
        data.all_targets = torch.cat(data.all_targets, dim=0)

    else:
        folder_path = data.dataDir
        files = [file for file in os.listdir(folder_path) if "train_inputs" in file]
        data.all_inputs = []
        data.all_targets = []
        data.all_vali_inputs = []
        data.all_vali_targets = []

        for file in files:
            train_inputs = torch.load(os.path.join(folder_path, file))
            train_targets = torch.load(os.path.join(folder_path, file.replace('train_inputs', 'train_targets')))
            vali_inputs = torch.load(os.path.join(folder_path, file.replace('train_inputs', 'vali_inputs')))
            vali_targets = torch.load(os.path.join(folder_path, file.replace('train_inputs', 'vali_targets')))

            data.all_inputs.append(train_inputs)
            data.all_targets.append(train_targets)
            data.all_vali_inputs.append(vali_inputs)
            data.all_vali_targets.append(vali_targets)

        data.all_inputs = torch.cat(data.all_inputs, dim=0)
        data.all_targets = torch.cat(data.all_targets, dim=0)
        data.all_vali_inputs = torch.cat(data.all_vali_inputs, dim=0)
        data.all_vali_targets = torch.cat(data.all_vali_targets, dim=0)


class SmartBedDataset(Dataset):
    TRAIN = 0
    TEST = 2

    def __init__(self, mode=TRAIN, dataDir="./dataset/saved_tensor_for_train",
                 dataDirTest="./dataset/saved_tensor_for_test"):
        if not (mode == self.TRAIN or mode == self.TEST):
            print("Error - SmartBedDataset invalid mode " + format(mode))
            exit(1)

        self.mode = mode
        self.dataDir = dataDir
        self.dataDirTest = dataDirTest  # only for mode==self.TEST

        TensorLoader(self, isTest=(mode == self.TEST))
        # print(self.all_inputs.shape)  # torch.Size([4063, 10, 12, 32, 64])

        self.totalLength = self.all_inputs.shape[0]
        print(f"Number of data loaded: {self.totalLength}")

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.all_inputs[idx], self.all_targets[idx]

    def denormalize(self, np_array):
        target_max = 60
        target_min = 0
        denormalized_data = np_array * (target_max - target_min) + target_min

        return denormalized_data


class ValiDataset(SmartBedDataset):
    def __init__(self, dataset):
        self.all_inputs = dataset.all_vali_inputs
        self.all_targets = dataset.all_vali_targets
        self.totalLength = self.all_inputs.shape[0]
        print(f"Number of data loaded: {self.totalLength}")

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.all_inputs[idx], self.all_targets[idx]


if __name__ == '__main__':
    data_set = SmartBedDataset()
    print(data_set[0][0].shape)  # data_set[0] means all_inputs, data_set[0][0] means all_inputs[0]

    dataset_length = len(data_set)
    print("Dataset length:", dataset_length)

    # Verify if the printed length matches the expected total length
    if dataset_length == data_set.totalLength:
        print("Length check passed!")

    vali_data_set = ValiDataset(data_set)
    print(vali_data_set[0][0].shape)

    vali_dataset_length = len(vali_data_set)
    print("Dataset length:", vali_dataset_length)

    # Verify if the printed length matches the expected total length
    if vali_dataset_length == vali_data_set.totalLength:
        print("Length check passed!")
