import torch
from torch.utils.data import Dataset
import os
import random

random.seed(21339)


class SmartBedDataset_Base(Dataset):
    TRAIN = 0
    TEST = 2

    def __init__(self, mode=TRAIN, dataDir="./dataset/saved_tensor_for_train",
                 dataDirTest="./dataset/saved_tensor_for_test", time_step=10,
                 input_mean=None, input_std=None, target_mean=None, target_std=None):
        if not (mode == self.TRAIN or mode == self.TEST):
            print("Error - SmartBedDataset invalid mode " + format(mode))
            exit(1)

        self.mode = mode

        if self.mode == self.TRAIN:
            self.folder_path = dataDir
        elif self.mode == self.TEST:
            self.folder_path = dataDirTest

        if input_mean is not None and input_std is not None and target_mean is not None and target_std is not None:
            self.input_mean = input_mean
            self.input_std = input_std
            self.target_mean = target_mean
            self.target_std = target_std
            self.all_inputs, self.all_targets, _, _, _, _ = self.TensorLoader()
        else:
            self.all_inputs, self.all_targets, \
                self.input_mean, self.input_std, self.target_mean, self.target_std = self.TensorLoader()
            torch.save(self.input_mean, 'dataset/saved_tensor_for_train/input_mean.pt')
            torch.save(self.input_std, 'dataset/saved_tensor_for_train/input_std.pt')
            torch.save(self.target_mean, 'dataset/saved_tensor_for_train/target_mean.pt')
            torch.save(self.target_std, 'dataset/saved_tensor_for_train/target_std.pt')

        self.length_sque = time_step
        self.length_each_file = [(file.shape[0] - self.length_sque + 1) for file in self.all_inputs]
        self.totalLength = sum(self.length_each_file)

        # print(f"Group number slicing by time step: {self.totalLength}")

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        file_idx = 0
        for length in self.length_each_file:
            if (idx + 1) <= length:  # idx begins at 0
                break
            else:
                file_idx += 1
                idx -= length
        data_now_input = self.all_inputs[file_idx]
        data_now_target = self.all_targets[file_idx]
        # print(data_now_input[idx: idx + self.length_sque].shape)

        normalized_input = (data_now_input - self.input_mean) / self.input_std
        normalized_target = (data_now_target - self.target_mean) / self.target_std

        return normalized_input[idx: idx + self.length_sque], normalized_target[idx + self.length_sque - 1]

    def TensorLoader(self):
        input_files_path = []
        target_files_path = []

        files = [file for file in os.listdir(self.folder_path) if "inputs" in file]
        for file in files:
            input_file_path = os.path.join(self.folder_path, file)
            target_file_path = os.path.join(self.folder_path, file.replace('inputs', 'targets'))
            input_files_path.append(input_file_path)
            target_files_path.append(target_file_path)
        # print(input_files_path)  # test code for file names

        all_inputs = [torch.load(file_name) for file_name in input_files_path]
        all_targets = [torch.load(file_name) for file_name in target_files_path]

        all_input_data = torch.cat(all_inputs, dim=0)
        input_mean = all_input_data.mean(dim=[0, 2, 3])
        input_std = all_input_data.std(dim=[0, 2, 3])

        input_std[input_std == 0] = 1e-8
        expanded_input_mean = input_mean.unsqueeze(1).unsqueeze(2)
        expanded_input_std = input_std.unsqueeze(1).unsqueeze(2)

        all_target_data = torch.cat(all_targets, dim=0)
        target_mean = all_target_data.mean()
        target_std = all_target_data.std()

        return all_inputs, all_targets, expanded_input_mean, expanded_input_std, target_mean, target_std

    def denormalize(self, np_array):
        denormalized_data = np_array * self.target_std.numpy() + self.target_mean.numpy()
        return denormalized_data


class SmartBedDataset_Train(SmartBedDataset_Base):
    # for divide train data into train and validation (70% data in one file for train)
    def __init__(self, ori_dataset, model):
        self.data_set = ori_dataset
        train_ratio = 0.7
        if not os.path.exists("dataset/saved_tensor_for_train/idx_train") and not os.path.exists(
                "dataset/saved_tensor_for_train/idx_vali"):
            print("Warning: Idx files not found. New files will be generated!!!")
            index = [i for i in range(len(ori_dataset))]
            random.shuffle(index)
            end_index = int(train_ratio * len(ori_dataset))
            train_idxs = index[0:end_index]
            vali_idxs = index[end_index:]
            torch.save(train_idxs, "dataset/saved_tensor_for_train/idx_train")
            torch.save(vali_idxs, "dataset/saved_tensor_for_train/idx_vali")
        else:
            train_idxs = torch.load("dataset/saved_tensor_for_train/idx_train")
            vali_idxs = torch.load("dataset/saved_tensor_for_train/idx_vali")

        if model == "train":
            self.idxs = train_idxs
        elif model == "validation":
            self.idxs = vali_idxs
        else:
            raise NotImplementedError("Wrong Model!")

        self.input_mean = ori_dataset.input_mean
        self.input_std = ori_dataset.input_std
        self.target_mean = ori_dataset.target_mean
        self.target_std = ori_dataset.target_std

        # test code
        # print(len(self.idxs))
        # print(self.idxs[0: 10])  # [38888, 23892, 8279, 33106, 46281, 7471, 24002, 40308, 18404, 2836]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.data_set[self.idxs[idx]]


class SmartBedDataset_Test(SmartBedDataset_Base):
    # for divide train data into train and validation (70% data in one file for train)
    def __init__(self, ori_dataset, model, input_mean, input_std, target_mean, target_std):
        self.data_set = ori_dataset

        if model == "test":
            pass
        else:
            raise NotImplementedError("Wrong Model!")

        self.input_mean = input_mean
        self.input_std = input_std
        self.target_mean = target_mean
        self.target_std = target_std

        self.totalLength = ori_dataset.totalLength

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.data_set[idx]


if __name__ == '__main__':
    raw_dataset = SmartBedDataset_Base()
    print(raw_dataset[0][0].shape)  # torch.Size([10, 12, 32, 64])
    print(raw_dataset[0][1].shape)  # torch.Size([1, 32, 64])

    dataset_length = len(raw_dataset)
    print("Dataset length:", dataset_length)

    # Verify if the printed length matches the expected total length
    if dataset_length == raw_dataset.totalLength:
        print()
        print("Length check passed!")

    train_dataset = SmartBedDataset_Train(raw_dataset, model="train")
    print(f"input_mean: {train_dataset.input_mean}")
    print(f"input_std: {train_dataset.input_std}")
    print(f"target_mean: {train_dataset.target_mean}")
    print(f"target_std: {train_dataset.target_std}")

    input_mean = torch.load('dataset/saved_tensor_for_train/input_mean.pt')
    input_std = torch.load('dataset/saved_tensor_for_train/input_std.pt')
    target_mean = torch.load('dataset/saved_tensor_for_train/target_mean.pt')
    target_std = torch.load('dataset/saved_tensor_for_train/target_std.pt')
    test_dataset = SmartBedDataset_Test(raw_dataset, model="test", input_mean=input_mean, input_std=input_std,
                                        target_mean=target_mean, target_std=target_std)

    print(test_dataset[0][0].shape)   # torch.Size([10, 12, 32, 64])
