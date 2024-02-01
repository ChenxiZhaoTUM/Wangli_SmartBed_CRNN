import os
import re
import scipy.stats
import matplotlib.pyplot as plt


def read_pressure():
    folder_path = "./files_for_correlation_analysis"
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
                    pressuremat_values = [int(value.strip()) for value in pressuremat_str.split(',')]

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
    plt.show()


if __name__ == "__main__":
    # data_dic = read_pressure()
    # print(data_dic)

    correlation_analysis()

