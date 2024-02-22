import os
import re
import numpy as np


def save_data_to_txt(file_path, data, suffix):
    save_file_path = file_path.replace(suffix, f"_Conver{suffix.capitalize()}")
    with open(save_file_path, "w") as file:
        for time, values in data.items():
            values_str = ", ".join(map(str, values))
            file.write(f"{time}: {values_str}\n")
    print(f"Data saved to {save_file_path}")


def process_file(file_path, is_origin):
    all_data = {}
    suffix = "_origin.txt" if is_origin else "_process.txt"

    with open(file_path, 'r', errors='ignore') as file:
        lines = file.readlines()

        if is_origin:
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

            for i in range(len(time_arr_txt)):
                time = time_arr_txt[i]
                all_data[time] = value_arr_txt[i]

        else:
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

            for i in range(len(times_process)):
                time = times_process[i]
                all_data[time] = values_process[i]

    save_data_to_txt(file_path, all_data, suffix)


def read_pressure(folder_path):
    file_names = os.listdir(folder_path)
    print("The operating files:")

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)

        if file_name.endswith("_origin.txt"):
            print(file_path)
            process_file(file_path, is_origin=True)

        elif file_name.endswith("_process.txt"):
            print(file_path)
            process_file(file_path, is_origin=False)


if __name__ == '__main__':
    folder_path = "./lowPres_check"
    read_pressure(folder_path)
