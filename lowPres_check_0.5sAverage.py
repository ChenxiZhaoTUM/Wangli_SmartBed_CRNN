import os
import re
from datetime import datetime
import numpy as np
import torch
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def parse_time_string(time_string):
    return datetime.strptime(time_string, "%H:%M:%S.%f")


def format_time_string(time):
    return time.strftime("%H:%M:%S")


def format_time_string_half_second(time):
    return time.strftime("%H:%M:%S.%f")


def round_to_nearest_half_second_down(time_obj):
    if time_obj.microsecond >= 500000:
        time_obj = time_obj.replace(microsecond=500000)
    else:
        time_obj = time_obj.replace(microsecond=0)

    return time_obj


def average_by_sec(required_time, time_arr, value_arr):
    time_objs = [parse_time_string(time_str) for time_str in time_arr]

    if required_time == 1:
        time_objs = [format_time_string(time_obj) for time_obj in time_objs]
    else:
        time_objs = [round_to_nearest_half_second_down(time_obj) for time_obj in time_objs]
        time_objs = [format_time_string_half_second(time_obj) for time_obj in time_objs]

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


def read_pressure():
    folder_path = "./lowPres_check"
    file_names = os.listdir(folder_path)

    all_process_data = {}
    all_origin_data = {}

    print("The operating files:")

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        print(file_path)

        with open(file_path, 'r', errors='ignore') as file:
            lines = file.readlines()

            if file_name.endswith("_process.txt"):
                times_process = []
                values_process = []

                for line in lines:
                    parts = line.strip().split('][')
                    if len(parts) == 3:
                        time_str = parts[0][1:]
                        airbag_str = parts[1]
                        pressuremat_str = parts[2][:-1]

                        airbag_values = [float(value.strip()) for value in airbag_str.split(',')]
                        pressuremat_values = [(4095 - int(value.strip())) for value in pressuremat_str.split(',')]

                        times_process.append(time_str)
                        values_process.append(pressuremat_values)

                times_process = np.array(times_process)
                values_process = np.array(values_process)

                avg_times_process, avg_values_process = average_by_sec(0.5, times_process, values_process)

                for i in range(len(avg_times_process)):
                    time = avg_times_process[i]
                    all_process_data[time] = avg_values_process[i]

            if file_name.endswith("_origin.txt"):
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

                avg_time_arr_txt, avg_value_arr_txt = average_by_sec(0.5, time_arr_txt, value_arr_txt)

                for i in range(len(avg_time_arr_txt)):
                    time = avg_time_arr_txt[i]
                    all_origin_data[time] = avg_value_arr_txt[i]

    return all_origin_data, all_process_data


def reshape_pressuremat_values(pressuremat_values):
    new_values = np.zeros((32, 64))
    for i in range(16):
        new_values[:, i * 4:(i + 1) * 4] = pressuremat_values[i]

    return new_values


def dynamic_pic():
    all_origin_data, all_process_data = read_pressure()

    common_data_in_single_file = {}
    for time, origin_data in all_origin_data.items():
        if time in all_process_data:
            process_data = all_process_data[time]

            common_data_id = len(common_data_in_single_file)
            common_data_in_single_file[common_data_id] = {
                'time': time,
                'origin_data': origin_data,
                'process_data': process_data
            }

    origin_values = np.zeros((len(common_data_in_single_file), 32, 64))
    process_values = np.zeros((len(common_data_in_single_file), 32, 64))

    for idx, data in common_data_in_single_file.items():
        origin_values[idx, :, :] = reshape_pressuremat_values(data['origin_data'])
        process_values[idx, :, :] = reshape_pressuremat_values(data['process_data'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.set_aspect('equal', 'box')
    origin_image = np.reshape(origin_values[0], (32, 64))
    im1 = ax1.imshow(origin_image, cmap='jet', interpolation='bilinear', vmin=-0, vmax=3000)
    ax1.axis('off')
    ax1.set_title("Original Values")
    fig.colorbar(im1, ax=ax1)

    ax2.set_aspect('equal', 'box')
    process_image = np.reshape(process_values[0], (32, 64))
    im2 = ax2.imshow(process_image, cmap='jet', interpolation='bilinear', vmin=-0, vmax=3000)
    ax2.axis('off')
    ax2.set_title("Pressed Values")
    fig.colorbar(im2, ax=ax2)

    def update(frame):
        target_frame = np.reshape(origin_values[frame], (32, 64))
        im1.set_array(target_frame)
        input_frame = np.reshape(process_values[frame], (32, 64))
        im2.set_array(input_frame)

    ani = animation.FuncAnimation(fig, update, frames=len(process_values), interval=100, blit=False)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # all_origin_data, all_process_data = read_pressure()
    #
    # common_data_in_single_file = {}
    # for time, origin_data in all_origin_data.items():
    #     if time in all_process_data:
    #         print(f"Time: {time}")
    #         print(f"Original Data: {all_origin_data[time]}")
    #         print(f"Processed Data: {all_process_data[time]}")
    #         print("--------------------------------------------------")

    dynamic_pic()
