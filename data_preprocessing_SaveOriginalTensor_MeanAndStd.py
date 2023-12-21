import os
import re
from datetime import datetime
import numpy as np
import torch


##### average values by second #####
def parse_time_string(time_string):
    return datetime.strptime(time_string, "%H:%M:%S.%f")


def format_time_string(time):
    return time.strftime("%H:%M:%S")


def average_by_sec(time_arr, value_arr):
    time_objs = [parse_time_string(time_str) for time_str in time_arr]
    time_objs = [format_time_string(time_obj) for time_obj in time_objs]

    # create dictionary to save time and corresponding values
    sum_dict = {}
    count_dict = {}
    new_time_arr = []

    for i in range(len(time_objs)):
        new_time_str = time_objs[i]

        if new_time_str in sum_dict:
            sum_dict[new_time_str] += value_arr[i]
            count_dict[new_time_str] += 1
        else:
            sum_dict[new_time_str] = value_arr[i]
            count_dict[new_time_str] = 1
            new_time_arr.append(new_time_str)

    avg_value_arr = [sum_dict[new_time_str] / count_dict[new_time_str] for new_time_str in new_time_arr]
    avg_value_arr = np.array(avg_value_arr)

    new_time_arr = np.array(new_time_arr)

    return new_time_arr, avg_value_arr


##### read output file #####
def deal_with_csv_file(csv_file_path):
    with open(csv_file_path, 'r', errors='ignore') as file:
        time_arr_csv = []
        value_arr_csv = []

        lines = file.readlines()[1:]

        for line in lines:
            line = line.strip()
            line = line.split(',')

            time_str = line[0].split(' ')[1]
            time_arr_csv.append(time_str)

            value_str = line[1:]
            value_per_time = [int(value) for value in value_str]
            value_arr_csv.append(value_per_time)

        time_arr_csv = np.array(time_arr_csv)
        value_arr_csv = np.array(value_arr_csv)
        return time_arr_csv, value_arr_csv


def reshape_output_value(value_arr):
    new_reshape_value_arr = []
    for value in value_arr:
        value = np.reshape(value, (32, 64))
        new_reshape_value_arr.append(value)

    new_reshape_value_arr = np.array(new_reshape_value_arr)
    return new_reshape_value_arr


##### read input files #####
def deal_with_txt_file(txt_file_path):
    with open(txt_file_path, 'r', errors='ignore') as file:
        lines = file.readlines()

        time_arr_txt = []
        value_arr_txt = []

        for line in lines:
            line = line.strip()

            # ignore the line not including []
            if "[" not in line and "]" not in line:
                continue

            time_start = line.find("[")
            time_end = line.find("]")

            if line[time_end + 1: time_end + 5] == "AA23" and line[-2:] == "55":
                pres_hex_values = line[time_end + 5: -2]
                if len(pres_hex_values) == 64:
                    processed_hex_values = ''.join(
                        [pres_hex_values[i + 2: i + 4] + pres_hex_values[i: i + 2] for i in
                         range(0, len(pres_hex_values), 4)])

                    pres_decimal_arr = [4095 - int(processed_hex_values[i:i + 4], 16) for i in
                                        range(0, len(processed_hex_values), 4)]
                else:
                    # print("Pressure values are less than 16.")
                    continue
            else:
                # print("Header/length/tail identification error.")
                continue

            time_str = line[time_start + 1: time_end]
            time_str = re.sub(r'[^a-zA-Z0-9:.]', '', time_str)
            time_arr_txt.append(time_str)
            value_arr_txt.append(pres_decimal_arr)

        time_arr_txt = np.array(time_arr_txt)
        value_arr_txt = np.array(value_arr_txt)
        return time_arr_txt, value_arr_txt


def deal_with_sleep_txt_file(sleep_txt_file_path):
    with open(sleep_txt_file_path, 'r', errors='ignore') as file:
        lines = file.readlines()

        time_arr_sleep_txt = []
        value_arr_sleep_txt = []

        for line in lines:
            line = line.strip()

            # ignore the line not including []
            if "[" not in line and "]" not in line:
                continue

            time_start = line.find("[")
            time_end = line.find("]")

            if line[time_end + 1: time_end + 5] == "AB11" and line[-2:] == "55":
                pres_hex_values = line[time_end + 5: -2]
                if len(pres_hex_values) == 28:
                    processed_hex_values = pres_hex_values[0: 12]

                    processed_hex_values = processed_hex_values + ''.join(
                        [pres_hex_values[i + 2: i + 4] + pres_hex_values[i: i + 2] for i in
                         range(12, 20, 4)])

                    processed_hex_values = processed_hex_values + pres_hex_values[20: 22]

                    processed_hex_values = processed_hex_values + ''.join(
                        [pres_hex_values[24: 26] + pres_hex_values[22: 24]])

                    processed_hex_values = processed_hex_values + pres_hex_values[26:]

                    pres_decimal_arr = [int(processed_hex_values[i:i + 2], 16) for i in range(0, 12, 2)]
                    pres_decimal_arr.extend([int(processed_hex_values[12:16], 16)])
                    pres_decimal_arr.extend([int(processed_hex_values[16:20], 16)])
                    pres_decimal_arr.append(int(processed_hex_values[20:22], 16))
                    pres_decimal_arr.append(int(processed_hex_values[22:26], 16))
                    pres_decimal_arr.append(int(processed_hex_values[26:28], 16))
                else:
                    # print("Pressure values are less than 14.")
                    continue
            else:
                # print("Header/length/tail identification error.")
                continue

            time_str = line[time_start + 1: time_end]
            time_str = re.sub(r'[^a-zA-Z0-9:.]', '', time_str)
            time_arr_sleep_txt.append(time_str)
            value_arr_sleep_txt.append(pres_decimal_arr)

        time_arr_sleep_txt = np.array(time_arr_sleep_txt)
        value_arr_sleep_txt = np.array(value_arr_sleep_txt)
        return time_arr_sleep_txt, value_arr_sleep_txt


pressureNormalization = True
inputNormalization = True


# change input_data to [12, 32, 64]
# change output_data to [1, 32, 64]
def change_dimension(input_data, input_sleep_data, target_data):
    new_input_data = torch.zeros(12, 32, 64)

    for ch in range(11):
        new_input_data[ch, :, :] = torch.tensor(input_sleep_data[ch])  # input_sleep_data to 11 channels

    for j in range(16):
        new_input_data[11, :, j * 4: (j + 1) * 4] = torch.tensor(input_data[j])

    new_target_data = torch.from_numpy(target_data).unsqueeze(0)
    return new_input_data, new_target_data


def loader_normalizer(inputs_pressure_arr, targets_arr):
    inputs_pressure_arr_norm = []
    targets_arr_norm = []

    if pressureNormalization:
        if len(targets_arr) > 0:
            # target_min = np.amin(targets_arr)
            # target_max = np.amax(targets_arr)
            target_min = 0
            target_max = 60

            for target_value in targets_arr:
                normalized_target_value = (target_value - target_min) / (target_max - target_min)
                targets_arr_norm.append(normalized_target_value)

            targets_arr = np.array(targets_arr_norm)

    if inputNormalization:
        if len(inputs_pressure_arr) > 0:
            for input_value in inputs_pressure_arr:
                normalized_input_data = input_value / 4096
                inputs_pressure_arr_norm.append(normalized_input_data)
            inputs_pressure_arr = np.array(inputs_pressure_arr_norm)

    return inputs_pressure_arr, targets_arr


##### save common data from file by file #####
def load_data_from_file(csv_file_path):
    filename_without_extension = os.path.splitext(os.path.basename(csv_file_path))[0]
    # print(f"The operating file: {filename_without_extension}")

    txt_file_path = os.path.join(os.path.dirname(csv_file_path), f"{filename_without_extension}.txt")
    if not os.path.isfile(txt_file_path):
        print(f"Error: {filename_without_extension}.txt does not exist.")
        exit(1)

    sleep_txt_file_path = os.path.join(os.path.dirname(csv_file_path), f"{filename_without_extension}Sleep.txt")
    if not os.path.isfile(sleep_txt_file_path):
        print(f"Error: {filename_without_extension}Sleep.txt does not exist.")
        exit(1)

    time_arr_txt, value_arr_txt = deal_with_txt_file(txt_file_path)
    time_arr_sleep_txt, value_arr_sleep_txt = deal_with_sleep_txt_file(sleep_txt_file_path)
    time_arr_csv, value_arr_csv = deal_with_csv_file(csv_file_path)

    # do time average
    avg_time_arr_txt, avg_value_arr_txt = average_by_sec(time_arr_txt, value_arr_txt)
    avg_time_arr_sleep_txt, avg_value_arr_sleep_txt = average_by_sec(time_arr_sleep_txt, value_arr_sleep_txt)
    avg_time_arr_csv, avg_value_arr_csv = average_by_sec(time_arr_csv, value_arr_csv)
    # avg_value_arr_txt, avg_value_arr_csv = loader_normalizer(avg_value_arr_txt, avg_value_arr_csv)

    avg_value_arr_csv = reshape_output_value(avg_value_arr_csv)

    ##### save common data from single file in dictionary #####
    inputs_in_single_file = {}
    inputs_sleep_in_single_file = {}
    targets_in_single_file = {}

    for i in range(len(avg_time_arr_txt)):
        time = avg_time_arr_txt[i]
        inputs_in_single_file[time] = avg_value_arr_txt[i]

    for i in range(len(avg_time_arr_sleep_txt)):
        time = avg_time_arr_sleep_txt[i]
        inputs_sleep_in_single_file[time] = avg_value_arr_sleep_txt[i]

    for i in range(len(avg_time_arr_csv)):
        time = avg_time_arr_csv[i]
        targets_in_single_file[time] = avg_value_arr_csv[i]

    common_data_in_single_file = {}
    for time, input_data in inputs_in_single_file.items():
        if time in inputs_sleep_in_single_file and time in targets_in_single_file:
            input_sleep_data = inputs_sleep_in_single_file[time]
            target_data = targets_in_single_file[time]

            new_input_data, new_target_data = change_dimension(input_data, input_sleep_data, target_data)

            common_data_id = len(common_data_in_single_file)
            common_data_in_single_file[common_data_id] = {
                'time': time,
                'input_data': new_input_data,
                'target_data': new_target_data
            }

    # print(f"Number of data loaded of {filename_without_extension}:", len(common_data_in_single_file))

    return common_data_in_single_file


# save input and target tensor of single file after preprocessing
def save_inputs_and_targets(isTest=False):
    if isTest:
        dataDir = "./dataset/for_test/"
    else:
        dataDir = "./dataset/for_train"
    csv_files_path = [os.path.join(dataDir, file) for file in os.listdir(dataDir) if file.endswith('.CSV')]

    for file_path in csv_files_path:
        common_data_in_single_file = load_data_from_file(file_path)

        data_length_in_single_file = len(common_data_in_single_file)
        inputs = []
        targets = []

        for common_data_id in range(data_length_in_single_file):
            value = common_data_in_single_file[common_data_id]
            input_data = value['input_data']
            target_data = value['target_data']
            inputs.append(input_data)
            targets.append(target_data)

        inputs = torch.stack(inputs)
        targets = torch.stack(targets)

        if isTest:
            saveDir = "./dataset/saved_tensor_for_test/"
            os.makedirs(saveDir, exist_ok=True)

        else:
            saveDir = "./dataset/saved_tensor_for_train/"
            os.makedirs(saveDir, exist_ok=True)

        inputs_save_path = os.path.join(saveDir,
                                        f"inputs_{os.path.splitext(os.path.basename(file_path))[0]}.pth")
        targets_save_path = os.path.join(saveDir,
                                         f"targets_{os.path.splitext(os.path.basename(file_path))[0]}.pth")

        torch.save(inputs, inputs_save_path)
        torch.save(targets, targets_save_path)

        print(inputs.shape)  # test code
        print(targets.shape)  # test code


if __name__ == '__main__':
    save_inputs_and_targets(isTest=False)
