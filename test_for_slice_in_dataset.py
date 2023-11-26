import numpy as np
import torch
import random
import os

random.seed(21339)


class dataset_base():

    def __init__(self, file_names_a=["./test_files/A1.txt", "./test_files/A2.txt"],
                 file_names_b=["./test_files/B1.txt", "./test_files/B2.txt"]):
        # self.data_a = [torch.load(file_name) for file_name in file_names_a]
        # self.data_b = [torch.load(file_name) for file_name in file_names_b]
        self.data_a = [self.load_text_data(file_name) for file_name in file_names_a]
        self.data_b = [self.load_text_data(file_name) for file_name in file_names_b]

        self.length_sque = 3
        self.length_each_file = [(file.shape[0] - self.length_sque + 1) for file in self.data_a]
        self.data_length = sum(self.length_each_file)

    def load_text_data(self, file_name):
        # Read the text file and parse the content
        with open(file_name, 'r') as file:
            lines = file.readlines()
            data = []
            for line in lines:
                values = [int(val) for val in line.strip().split(',')]
                data.append(values)
        return torch.tensor(data)

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        print(self.data_length)
        file_idx = 0
        for length in self.length_each_file:
            if (idx + 1) <= length:
                break
            else:
                file_idx += 1
                idx -= length
        data_now_a = self.data_a[file_idx]
        data_now_b = self.data_b[file_idx]
        print(data_now_a[idx: idx + self.length_sque])
        # print(data_now_a[idx: idx + self.length_sque].shape)  # torch.Size([3, 2])
        return data_now_a[idx: idx + self.length_sque], data_now_b[idx + self.length_sque - 1]


class data_set_run():

    def __init__(self, ori_dataset, model):
        self.data_set = ori_dataset
        train_ratio = 0.7
        if not os.path.exists("./test_files/idx_train") and not os.path.exists("./test_files/idx_vali"):
            print("Warning: Idx files not found. New files will be generated!!!")
            index = [i for i in range(len(ori_dataset))]
            random.shuffle(index)
            end_index = int(train_ratio * len(ori_dataset))
            train_idxs = index[0:end_index]
            vali_idxs = index[end_index:]
            torch.save(train_idxs, "./test_files/idx_train")
            torch.save(vali_idxs, "./test_files/idx_vali")

        else:
            train_idxs = torch.load("./test_files/idx_train")
            vali_idxs = torch.load("./test_files/idx_vali")

        if model == "train":
            self.idxs = train_idxs
        elif model == "vali":
            self.idxs = vali_idxs
        else:
            raise NotImplementedError("Wrong Model!")

        # print(len(self.idxs))  # 16 * 0.7 = 11
        # print(self.idxs)  # [3, 0, 5, 11, 14, 10, 4, 12, 15, 6, 9]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.data_set[self.idxs[idx]]


raw_dataset = dataset_base()
# print(raw_dataset[11])  # 12th group: (tensor([[2, 1], [2, 2], [2, 3]]), tensor([2, 3]))
print(raw_dataset[9])  # 10th group: (tensor([[ 1,  9], [ 1, 10], [ 1, 11]]), tensor([ 1, 11]))
train_dataset = data_set_run(raw_dataset, model="train")
