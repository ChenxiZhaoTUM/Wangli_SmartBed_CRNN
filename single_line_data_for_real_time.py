import datetime as datetime
import os
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from model_CRNN import CRNN

# pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cpu
use_cuda = torch.cuda.is_available()  # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

torch.set_printoptions(edgeitems=3, linewidth=200, precision=10, sci_mode=False, threshold=5000)


# 处理原始的16进制低精度压力数据，提取时间和对应的16组压力数据
def process_pressure_values(line):
    time_match = re.search(r'\[(.*?)\]', line)
    if time_match:
        time = time_match.group(1)
        values = line.split(']')[-1]

        if values.startswith("AA23") and values.endswith("55"):
            pres_hex_values = values[4:-2]

            if len(pres_hex_values) == 64:
                processed_hex_values = ''.join(
                    [pres_hex_values[i + 2: i + 4] + pres_hex_values[i: i + 2] for i in
                     range(0, len(pres_hex_values), 4)])

                pres_decimal_arr = [4095 - int(processed_hex_values[i:i + 4], 16) for i in
                                    range(0, len(processed_hex_values), 4)]

                return time, pres_decimal_arr

    return time, []


# 处理原始的睡眠数据
def process_sleep_values(line):
    time_match = re.search(r'\[(.*?)\]', line)
    if time_match:
        time = time_match.group(1)
        values = line.split(']')[-1]

        if values.startswith("AB11") and values.endswith("55"):
            pres_hex_values = values[4:-2]

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

                return time, pres_decimal_arr

    return time, []


# change input dimension to [12, 32, 64]
def change_input_dimension(input_data_arr, input_sleep_data_arr):
    new_input_data = torch.zeros(12, 32, 64)

    for ch in range(11):
        new_input_data[ch, :, :] = torch.tensor(input_sleep_data_arr[ch])  # input_sleep_data to 11 channels

    for j in range(16):
        new_input_data[11, :, j * 4: (j + 1) * 4] = torch.tensor(input_data_arr[j])

    return new_input_data


# mean and std for normalization
input_mean = torch.load('model_for_realtime/input_mean.pt')
input_std = torch.load('model_for_realtime/input_std.pt')
target_mean = torch.load('model_for_realtime/target_mean.pt')
target_std = torch.load('model_for_realtime/target_std.pt')


def input_normalization(input_tensor):
    normalized_input = (input_tensor - input_mean) / input_std
    return normalized_input


def output_denormalization(np_array):
    denormalized_output = np_array * target_std.numpy() + target_mean.numpy()

    output_clipped = np.clip(denormalized_output, a_min=0, a_max=None)
    return output_clipped


# List to store individual tensors, maintaining a rolling window of ten tensors
global_tensor_list = []


def result_of_CRNN(model, avg_pressure, sleep_value):
    global global_tensor_list

    input_tensor = change_input_dimension(avg_pressure, sleep_value)
    normalized_input = input_normalization(input_tensor)

    # If the list already has ten tensors, remove the first one
    if len(global_tensor_list) >= 10:
        global_tensor_list.pop(0)

    # Add the new tensor to the list
    global_tensor_list.append(normalized_input)
    print(f"Input list length: {len(global_tensor_list)}")

    if len(global_tensor_list) == 10:
        combined_tensor = torch.stack(global_tensor_list, dim=0)  # [10, 12, 32, 64]
        # print(combined_tensor[:, :, 1, 1])
        combined_tensor = combined_tensor.clone().unsqueeze(0)  # [1, 10, 12, 32, 64]
        # print(f"Combined tensor shape: {combined_tensor.shape}")

        if torch.all(combined_tensor[:, :, 11, :, :] == 0):  # if input_pressure = 0, i.e. nobody on bed
            output = torch.zeros(1, 32, 64)
        else:
            output = model(combined_tensor)
            output[output < 0] = 0

        # print(f"Output tensor shape: {output.shape}")
        denormalized_output = output_denormalization(output.detach().numpy())

        return combined_tensor, denormalized_output

    return None


# load trained NN
def load_model():
    # load trained CRNN model
    output_dir = "./TEST"
    os.makedirs(output_dir, exist_ok=True)

    netG = CRNN(channelExponent=4, dropout=0.0)
    doLoad = "model_for_realtime/CRNN_expo4_mean_04_03_2000model"
    # doLoad = "model_for_realtime/CRNN_expo4_mean_04_02_5000model"
    if len(doLoad) > 0:
        netG.load_state_dict(torch.load(doLoad, map_location=torch.device('cpu')))
    netG.to(device)
    netG.eval()
    return netG


# animation display
def LowPressureData2img(model, pressure_lines, sleep_line):
    sum_arr = np.zeros(16)
    num = 0
    for pressure_line in pressure_lines:
        pressure_time, pressure_value = process_pressure_values(pressure_line)
        sum_arr += pressure_value
        num += 1
    avg_pressure = sum_arr / num

    print(avg_pressure)

    sleep_time, sleep_value = process_sleep_values(sleep_line)

    result = result_of_CRNN(model, avg_pressure, sleep_value)

    if result is not None:
        input_tensor, output_denormalize = result
        print(f"Output tensor shape: {output_denormalize.shape}")
        print("Now start to output image!")
        print()

        output_image = np.reshape(output_denormalize[0], (32, 64))
        normalized_image = cv2.normalize(output_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        color_image = cv2.applyColorMap(normalized_image, cv2.COLORMAP_JET)
        resized_color_image = cv2.resize(color_image, (640, 320), interpolation=cv2.INTER_CUBIC)
        # smoothed_image = cv2.GaussianBlur(color_image, (3, 3), 0)
        # smoothed_image = cv2.bilateralFilter(color_image, 9, 75, 75)
        cv2.imshow('pressure distribution', resized_color_image)

        os.chdir("./TEST/")
        current_time = datetime.datetime.now()
        strdispaly = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        cv2.imwrite(strdispaly + "_pressure_distribution.png", resized_color_image)
        os.chdir("../")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return


# save images
def imageOut(filename, _input, _output, max_val=40, min_val=0):
    output = np.copy(_output)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    last_channel = _input[-1, -1, :, :]
    last_channel = last_channel * input_std[-1] + input_mean[-1]
    last_channel = np.delete(last_channel, [4 * i + 3 for i in range(16)], axis=1)
    last_channel = np.concatenate((last_channel, np.zeros((32, 16))), axis=1)
    last_channel_image = np.reshape(last_channel, (32, 64))
    ax1.set_aspect('equal', 'box')
    im1 = ax1.imshow(last_channel_image, cmap='jet', vmin=0, vmax=2500)
    ax1.axis('off')
    cbar1 = fig.colorbar(im1, ax=ax1)

    ax2.set_aspect('equal', 'box')
    output_image = np.reshape(output, (32, 64))
    im2 = ax2.imshow(output_image, cmap='jet', vmin=min_val, vmax=max_val)
    ax2.axis('off')
    cbar2 = fig.colorbar(im2, ax=ax2)

    plt.tight_layout()
    save_path = os.path.join(filename)
    plt.savefig(save_path)
    plt.close(fig)
