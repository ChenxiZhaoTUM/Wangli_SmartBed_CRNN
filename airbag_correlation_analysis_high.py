import os
import re
from datetime import datetime
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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


def deal_with_txt_file(txt_file_path):
    with open(txt_file_path, 'r', errors='ignore') as file:
        time_arr_txt = []
        value_arr_txt = []

        lines = file.readlines()

        for line in lines:
            parts = line.strip().split('][')
            if len(parts) == 3:
                time_str = parts[0][1:]
                airbag_str = parts[1]
                pressuremat_str = parts[2][:-1]

                airbag_values = [float(value.strip()) for value in airbag_str.split(',')]
                pressuremat_values = [int(value.strip()) for value in pressuremat_str.split(',')]

                time_arr_txt.append(time_str)
                value_arr_txt.append(airbag_values)

        time_arr_txt = np.array(time_arr_txt)
        value_arr_txt = np.array(value_arr_txt)
        return time_arr_txt, value_arr_txt


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


def read_data():
    folder_path = "./files_for_correlation_analysis/high"
    file_names = os.listdir(folder_path)

    all_common_data = {}
    print()
    print("The operating files:")

    for file_name in file_names:
        csv_file_path = os.path.join(folder_path, file_name)
        if csv_file_path.endswith(".CSV"):
            file_name_without_extension = os.path.splitext(file_name)[0]

            txt_file_name = file_name_without_extension + ".txt"
            txt_file_path = os.path.join(folder_path, txt_file_name)
            if not os.path.isfile(txt_file_path):
                continue

            print(file_name_without_extension)  # print available files

            time_arr_txt, value_arr_txt = deal_with_txt_file(txt_file_path)
            time_arr_csv, value_arr_csv = deal_with_csv_file(csv_file_path)

            # do time average
            avg_time_arr_txt, avg_value_arr_txt = average_by_sec(time_arr_txt, value_arr_txt)
            avg_time_arr_csv, avg_value_arr_csv = average_by_sec(time_arr_csv, value_arr_csv)

            all_airbag_data = {}
            all_mat_data = {}

            for i in range(len(avg_time_arr_txt)):
                time = avg_time_arr_txt[i]
                all_airbag_data[time] = avg_value_arr_txt[i]

            for i in range(len(avg_time_arr_csv)):
                time = avg_time_arr_csv[i]
                all_mat_data[time] = avg_value_arr_csv[i]

            for time, airbag_data in all_airbag_data.items():
                if time in all_mat_data:
                    mat_data = all_mat_data[time]
                    common_data_id = len(all_common_data)
                    all_common_data[common_data_id] = {
                        'time': time,
                        'airbag_data': airbag_data,
                        'mat_data': mat_data
                    }

    print()
    print("Number of data loaded:", len(all_common_data))

    return all_common_data


def correlation_analysis():
    data_dic = read_data()
    correlation_matrix = []

    for i in range(6):
        correlations = []
        for j in range(2048):
            airbag_value = [data['airbag_data'][i] for data in data_dic.values()]
            mat_value = [data['mat_data'][j] for data in data_dic.values()]

            # check airbag_value and mat_value is constant or not
            if np.std(airbag_value) == 0 or np.std(mat_value) == 0:
                corr = np.nan
            else:
                # Pearson correlation coefficient (linear) or Spearman's rank correlation coefficient (classify)
                corr, _ = scipy.stats.pearsonr(airbag_value, mat_value)
            correlations.append(corr)

        correlation_matrix.append(correlations)

    for row in correlation_matrix:
        print(row)

    return np.array(correlation_matrix)


def corretion_display():
    correlation_matrix = correlation_analysis()
    for i, correlations in enumerate(correlation_matrix):
        reshaped_correlations = correlations.reshape((32, 64))
        plt.figure(figsize=(10, 5))
        plt.imshow(reshaped_correlations, cmap='rainbow', interpolation='nearest', vmin=-0.5, vmax=0.5)
        plt.colorbar()
        plt.title(f'Correlation Airbag {i + 1} for highPresMat')
        plt.savefig(f'./files_for_correlation_analysis/high/airbag_{i + 1}_highPresMat_correlation.png')
        plt.show()
        plt.close()


def corretion_display_3D():
    correlation_matrix = correlation_analysis()

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    z_offsets = np.linspace(0, 5, 6)

    x = np.arange(64)
    y = np.arange(32)
    X, Y = np.meshgrid(x, y)

    for i, matrix in enumerate(correlation_matrix):
        Z = matrix.flatten()
        Z_pos = np.full(Z.shape, z_offsets[i])

        img = ax.scatter(X.flatten(), Y.flatten(), Z_pos, c=Z, cmap='rainbow', label=f'Airbag {i + 1}')

    ax.legend()
    ax.set_xlabel('X axis (Pressure Mat Point)')
    ax.set_ylabel('Y axis (Pressure Mat Point)')
    ax.set_zlabel('Z axis (Airbag)')

    fig.colorbar(img, shrink=0.5, aspect=5, label='Correlation Coefficient')

    plt.show()


def dynamic_pic(airbag_values, mat_values):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 2]})
    plt.subplots_adjust(left=0.2, right=0.8, top=0.95, bottom=0.05)

    x = np.arange(1, 7)
    line, = ax1.plot([], [], '-o')
    ax1.set_xlim(-2, 10)
    ax1.set_xticks(np.arange(1, 7))
    ax1.set_ylim(np.min(airbag_values), np.max(airbag_values))
    ax1.legend()
    ax1.set_title("Airbag Pressure Values")
    ax1.set_xlabel("Airbag Number")
    ax1.set_ylabel("Pressure Value")

    mat_image = np.reshape(mat_values[0], (32, 64))
    im2 = ax2.imshow(mat_image, cmap='jet', interpolation='bilinear')
    ax2.axis('off')
    ax2.set_title("Pressure Mat Values")
    fig.colorbar(im2, ax=ax2)

    def update(frame):
        y = airbag_values[frame, :]
        line.set_data(x, y)

        im2.set_array(mat_values[frame])

    ani = animation.FuncAnimation(fig, update, frames=len(mat_values), interval=100, blit=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # corretion_display()

    data_dic = read_data()

    airbag_values = np.zeros((len(data_dic), 6))
    mat_values = np.zeros((len(data_dic), 32, 64))

    for idx, data in data_dic.items():
        airbag_values[idx, :] = data['airbag_data']
        mat_values[idx, :, :] = np.reshape(data['mat_data'], (32, 64))

    dynamic_pic(airbag_values, mat_values)
