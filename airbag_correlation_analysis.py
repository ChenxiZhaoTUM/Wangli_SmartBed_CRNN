import os
import re
from datetime import datetime
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def read_pressure():
    folder_path = "./files_for_correlation_analysis/low"
    file_names = os.listdir(folder_path)

    data_dic = {}
    id = 0
    print("The operating files:")

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        print(file_path)

        with open(file_path, 'r', errors='ignore') as file:
            lines = file.readlines()

            for line in lines:
                parts = line.strip().split('][')
                if len(parts) == 3:
                    time_str = parts[0][1:]
                    airbag_str = parts[1]
                    pressuremat_str = parts[2][:-1]

                    airbag_values = [float(value.strip()) for value in airbag_str.split(',')]
                    pressuremat_values = [(4095 - int(value.strip())) for value in pressuremat_str.split(',')]

                    data_dic[id] = {
                        'time': time_str,
                        'airbag_values': airbag_values,
                        'pressuremat_values': pressuremat_values
                    }

                    id += 1

    return data_dic


def correlation_analysis():
    data_dic = read_pressure()
    correlation_matrix = []

    for i in range(6):
        correlations = []
        for j in range(16):
            airbag_value = [data['airbag_values'][i] for data in data_dic.values()]
            pressuremat_value = [data['pressuremat_values'][j] for data in data_dic.values()]

            # Pearson correlation coefficient (linear) or Spearman's rank correlation coefficient (classify)
            corr, _ = scipy.stats.pearsonr(airbag_value, pressuremat_value)
            correlations.append(corr)

        correlation_matrix.append(correlations)

    for row in correlation_matrix:
        print(row)

    for i, correlations in enumerate(correlation_matrix):
        plt.plot(range(1, 17), correlations, label=f'Airbag {i + 1}')

    plt.xlabel('Pressure Mat Point')
    plt.ylabel('Correlation Coefficient')
    plt.title('Airbag vs Pressure Mat Correlation')
    plt.legend()
    plt.savefig(f'./files_for_correlation_analysis/low/airbag_lowPresMat_correlation.png')
    plt.show()
    plt.close()


def reshape_pressuremat_values(pressuremat_values):
    new_values = np.zeros((32, 64))
    for i in range(16):
        new_values[:, i * 4:(i + 1) * 4] = pressuremat_values[i]

    return new_values

def dynamic_pic():
    data_dic = read_pressure()

    airbag_values = np.zeros((len(data_dic), 6))
    mat_values = np.zeros((len(data_dic), 32, 64))

    for idx, data in data_dic.items():
        airbag_values[idx, :] = data['airbag_values']
        mat_values[idx, :, :] = reshape_pressuremat_values(data['pressuremat_values'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 2]})
    plt.subplots_adjust(left=0.2, right=0.8, top=0.95, bottom=0.05)

    x = np.arange(1, 7)
    line, = ax1.plot([], [], '-o', label='Airbag Pressure')
    ax1.set_xlim(-2, 8)
    ax1.set_xticks(np.arange(1, 7))
    ax1.set_ylim(np.min(airbag_values), np.max(airbag_values))
    ax1.legend()
    ax1.set_title("Airbag Pressure Values")
    ax1.set_xlabel("Airbag Number")
    ax1.set_ylabel("Pressure Value")

    mat_image = np.reshape(mat_values[0], (32, 64))
    im2 = ax2.imshow(mat_image, cmap='jet', interpolation='bilinear', vmin=-0, vmax=2000)
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
    # data_dic = read_pressure()
    # print(data_dic)

    # correlation_analysis()

    dynamic_pic()
