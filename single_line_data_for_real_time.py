import re
import math, os
import torch
import numpy as np
from model_CRNN import CRNN
import matplotlib.pyplot as plt


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


def process_sleep_values(line):
    time_match = re.search(r'\[(.*?)\]', line)
    if time_match:
        time = time_match.group(1)
        values = line.split(']')[-1]

        if values.startswith("AB11") and values.endswith("55"):
            pres_hex_values = values[4:-2]

            if len(pres_hex_values) == 28:
                processed_hex_values = pres_hex_values[:12] + ''.join(
                    [pres_hex_values[i + 2:i + 4] + pres_hex_values[i:i + 2] for i in range(12, 20, 4)])
                processed_hex_values += pres_hex_values[20:22] + pres_hex_values[24:26] + \
                                        pres_hex_values[22:24] + pres_hex_values[26:28]

                pres_decimal_arr = [int(processed_hex_values[i:i + 2], 16) for i in range(0, 12, 2)] + [
                    int(processed_hex_values[i:i + 4], 16) for i in range(12, 24, 4)] + [
                                       int(processed_hex_values[24:26], 16), int(processed_hex_values[26:28], 16)]
                return time, pres_decimal_arr

    return time, []


def change_dimension(input_data_arr, input_sleep_data_arr):
    input_data_arr_norm = []
    if len(input_data_arr) > 0:
        for input_value in input_data_arr:
            normalized_input_data = input_value / 4096
            input_data_arr_norm.append(normalized_input_data)
        input_data_arr_norm = np.array(input_data_arr_norm)

    new_input_data = torch.zeros(12, 32, 64)

    for ch in range(11):
        new_input_data[ch, :, :] = torch.tensor(input_sleep_data_arr[ch])  # input_sleep_data to 11 channels

    for j in range(16):
        new_input_data[11, :, j * 4: (j + 1) * 4] = torch.tensor(input_data_arr_norm[j])

    return new_input_data


def denormalize(np_array):
    target_max = 60
    target_min = 0
    denormalized_data = np_array * (target_max - target_min) + target_min

    output_clipped = np.clip(denormalized_data, a_min=0, a_max=None)
    return output_clipped


def print_3d_vector(vec):
    for i, channel in enumerate(vec):
        print(f"Channel {i}:")
        for row in channel:
            print(" ".join(map(str, row)))
        print()


def imageOut(filename, _input, _output, max_val=40, min_val=0):
    output = np.copy(_output)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    last_channel = _input[-1, -1, :, :]
    last_channel = np.delete(last_channel, [4 * i + 3 for i in range(16)], axis=1)
    last_channel = np.concatenate((last_channel, np.zeros((32, 16))), axis=1)
    last_channel_image = np.reshape(last_channel, (32, 64))
    ax1.set_aspect('equal', 'box')
    im1 = ax1.imshow(last_channel_image, cmap='jet', vmin=0, vmax=0.6)
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


def test01():
    line0 = "[16:07:59.569]AA23F40FFF0FFF0F3B0FFF0FFF0FFF0F8E0F280D64086B0CBC05140B7F0C4F0AFF0F55"
    time, values = process_pressure_values(line0)
    # print(values)
    pressure_map = {time: values}

    line0_sleep = "[16:07:59.789]AB11450C00000001861B060000710A1955"
    time_sleep, values_sleep = process_sleep_values(line0_sleep)
    sleep_map = {time_sleep: values_sleep}

    new_input_data_tensor = change_dimension(values, values_sleep)
    print(new_input_data_tensor)
    # print_3d_vector(new_input_data)


def test02():
    output_dir = "TEST"
    os.makedirs(output_dir, exist_ok=True)

    netG = CRNN(channelExponent=4, dropout=0.0)
    doLoad = "./CRNN_expo4_02_model"
    if len(doLoad) > 0:
        netG.load_state_dict(torch.load(doLoad, map_location=torch.device('cpu')))
    netG.eval()

    # layer1 = netG.layer1
    # layer1_conv = layer1.layer1_conv
    # print("Parameters of layer 'layer1_conv':")
    # for param in layer1_conv.parameters():
    #     print(param.size())
    #     print(param.data)

    # pressure_lines = [
    #     "[16:07:59.569]AA23F40FFF0FFF0F3B0FFF0FFF0FFF0F8E0F280D64086B0CBC05140B7F0C4F0AFF0F55",
    #     "[16:08:00.570]AA23EA0FFF0FFF0F3F0FFF0FFF0FFF0FFF0F1B0D5D08680CB405FE0A9E0C6F0AFF0F55",
    #     "[16:08:01.567]AA23EC0FFF0FFF0F3C0FFF0FFF0FFF0FFC0F240D6E084E0CB905FA0AB40C4C0AFF0F55",
    #     "[16:08:02.564]AA23F40FFF0FFF0F3D0FFF0FFF0FFF0FF60FF70C4508650CBF05290BB30C6F0ABC0F55",
    #     "[16:08:03.577]AA23EF0FFF0FFF0F3F0FFF0FFF0FFF0FF30F280D3808550CBB05250BB90C5E0AAF0F55",
    #     "[16:08:04.565]AA23F50FFF0FFF0F450FFF0FFF0FFF0FF90F2F0D3D08580CB305090B660D6B0AFF0F55",
    #     "[16:08:05.578]AA23F40FFF0FFF0F390FFF0FFF0FFF0FF70F330D2508560CAC05ED0AA90C0D0AFF0F55",
    #     "[16:08:06.577]AA23EF0FFF0FFF0F380FFF0FFF0FFF0FEC0F650D23085C0CA505FD0A890C5F0AFC0F55",
    #     "[16:08:07.575]AA23F50FFF0FFF0F370FFF0FFF0FFF0FF30F7F0D0B08590CA7051D0B7D0C330AFF0F55",
    #     "[16:08:08.573]AA23F20FFF0FFF0F380FFF0FFF0FFF0FF40FA30D1508570CA405FE0A930C570AE80F55",
    #     "[16:08:09.576]AA23F40FFF0FFF0F330FFF0FFF0FFF0FF20F7F0D2508540C9F05D80A7B0C580AA30F55",
    #     "[16:08:10.577]AA23F30FFF0FFF0F390FFF0FFF0FFF0FF30FB00D2408480C9F05E90A850C160A9D0F55",
    #     "[16:08:11.576]AA23F40FFF0FFF0F330FFF0FFF0FFF0FF00F740D1B086F0C9C05150B630C440AA50F55"
    # ]

    # pressure_lines = [
    #     "[16:07:59.569]AA23F40FFF0FFF0F3B0FFF0FFF0FFF0F8E0F280D64086B0CBC05140B7F0C4F0AFF0F55",
    #     "[16:07:59.678]AA23F60FFF0FFF0F470FFF0FFF0FFF0F860FFF0C6F086C0CBB050B0BA30C570AFF0F55",
    #     "[16:07:59.774]AA23F50FFF0FFF0F470FFF0FFF0FFF0F9A0F1B0D68086F0CB905080B880C5C0AFF0F55",
    #     "[16:07:59.867]AA23F80FFF0FFF0F470FFF0FFF0FFF0F870F0B0D5F086C0CB505090B7C0C6A0AFF0F55",
    #     "[16:07:59.977]AA23EF0FFF0FFF0F450FFF0FFF0FFF0F8B0F240D5F08670CB4050B0B770C610AFF0F55",
    #     "[16:08:00.070]AA23EE0FFF0FFF0F3B0FFF0FFF0FFF0F8B0F1F0D65086A0CB505000B670C650AFF0F55",
    #     "[16:08:00.165]AA23F30FFF0FFF0F470FFF0FFF0FFF0F9E0F270D5F086B0CB405050B6B0C690AFF0F55",
    #     "[16:08:00.273]AA23F30FFF0FFF0F3B0FFF0FFF0FFF0FFF0F2D0D5B086D0CB405FD0A690C6B0AFF0F55",
    #     "[16:08:00.366]AA23F50FFF0FFF0F450FFF0FFF0FFF0FFF0F1D0D57086B0CB505FE0A9F0C6B0AFF0F55",
    #     "[16:08:00.476]AA23F30FFF0FFF0F3F0FFF0FFF0FFF0FFC0F0B0D5808690CB505FB0A9D0C630AFF0F55",
    #     "[16:08:00.570]AA23EA0FFF0FFF0F3F0FFF0FFF0FFF0FFF0F1B0D5D08680CB405FE0A9E0C6F0AFF0F55",
    #     "[16:08:00.664]AA23EF0FFF0FFF0F440FFF0FFF0FFF0FFF0F1E0D63086C0CB305FF0AAF0C760AFF0F55",
    #     "[16:08:00.774]AA23EF0FFF0FFF0F450FFF0FFF0FFF0FFF0F2F0D6508690CB505F70AA70C870AFF0F55"
    # ]
    #
    # sleep_lines = [
    #     "[16:07:59.789]AB11450C00000001861B060000710A1955",
    #     "[16:08:00.790]AB11440D01010001DD190600006E0A1455",
    #     "[16:08:01.800]AB11440D01010001CA180700006C0A1455",
    #     "[16:08:02.797]AB11440D010100017E18070000700A1355",
    #     "[16:08:03.796]AB11440D01010001D7180800006E0A1755",
    #     "[16:08:04.810]AB11440D0301000122190900006E0A1955",
    #     "[16:08:05.812]AB11450D0301000171190A00006C0A1755",
    #     "[16:08:06.824]AB11450D0101000172190A00006E0A1955",
    #     "[16:08:07.815]AB11460D0000000126190B00006E0A1655",
    #     "[16:08:09.810]AB11450D0000000158170B00006B0A1055",
    #     "[16:08:10.815]AB11440D00000001BD160B00006A0A1155",
    #     "[16:08:11.824]AB11430D000000014C160B00006B0A1A55"
    # ]

    pressure_lines = [
        "[13:37:28.297]AA23FF0F4F0DFF0FB70F110FFF0FFF0FA30EFF0FDD093B08AF05380E140C1B08680E55",
        "[13:37:28.390]AA23FF0F470DFF0FD30F0C0FFF0FFF0F9F0EFF0FDF093B08B305330E1D0C1A08690E55",
        "[13:37:28.484]AA23FF0F4B0DFF0FD10F130FFF0FFF0FA60EFF0FDB093A08AD05450E1F0C18086F0E55",
        "[13:37:28.593]AA23FF0F570DFF0FC60F0F0FFF0FFF0FA30EFF0FE3093E08B505350E110C0F081C0E55",
        "[13:37:28.687]AA23FF0F530DFF0FB50F0F0FFF0FFF0FA30EFF0FE7094908AF052C0E050C0708680E55",
        "[13:37:28.780]AA23FF0F4F0DFF0FDD0F130FFF0FFF0F9F0EFF0FE5093E08B5053C0E1F0C0508440E55",
        "[13:37:28.888]AA23FF0F550DFF0FDB0F120FFF0FFF0F9C0EFF0FE3093708B4052D0E1F0CFF07170E55",
        "[13:37:28.982]AA23FF0F4A0DFF0FCF0F0D0FFF0FFF0F9C0EFF0FE5093D08B9052F0E2E0C07082D0E55",
        "[13:37:29.092]AA23FF0F3B0DFF0FD30F090FFF0FFF0F9C0EFF0FDF093708B405380E1C0C1208350E55",
        "[13:37:29.185]AA23FF0F3F0DFF0FD60F0B0FFF0FFF0FA30EFF0FE5092D08B5053F0E0F0C1C08340E55",
        "[13:37:29.279]AA23FF0F3D0DFF0FC70F0D0FFF0FFF0FA70EFF0FDD092B08B305370EFD0B1F083D0E55",
        "[13:37:29.388]AA23FF0F3C0DFF0FBD0F0B0FFF0FFF0F9B0EFF0FE3092E08AE052F0E110C0B08330E55",
        "[13:37:29.482]AA23FF0F2D0DFF0FBF0F0D0FFF0FFF0FA40EFF0FD7092F08AE052F0E0B0C0B08370E55"
    ]

    sleep_lines = [
        "[13:37:28.530]AB11000000000000FC1300000097093455",
        "[13:37:29.545]AB110000000000000E1400000099093455",
        "[13:37:30.538]AB11000000000000E31300000097093455",
        "[13:37:31.539]AB110000000000002C1400000097093955"
    ]

    pressure_data = {process_pressure_values(line)[0]: process_pressure_values(line)[1] for line in pressure_lines}
    sleep_data = {process_sleep_values(line)[0]: process_sleep_values(line)[1] for line in sleep_lines}

    # List to store individual tensors, maintaining a rolling window of ten tensors
    tensor_list = []
    i = 0

    for pressure_time in pressure_data:
        pressure_seconds = pressure_time.split(':')[2].split('.')[0]
        for sleep_time in sleep_data:
            if sleep_time.split(':')[2].split('.')[0] == pressure_seconds:
                input_tensor = change_dimension(pressure_data[pressure_time], sleep_data[sleep_time])

                # If the list already has ten tensors, remove the first one
                if len(tensor_list) == 10:
                    tensor_list.pop(0)

                # Add the new tensor to the list
                tensor_list.append(input_tensor)

                if len(tensor_list) == 10:
                    # Combine them into a single tensor with shape [10, 12, 32, 64]
                    combined_tensor = torch.stack(tensor_list, dim=0)  # Using numpy for stacking
                    # print(f"Combined tensor shape: {combined_tensor.shape}")
                    combined_tensor = combined_tensor.clone().unsqueeze(0)  # [1, 10, 12, 32, 64]
                    output = netG(combined_tensor)
                    output_denormalize = denormalize(output.detach().numpy())
                    # print(output_denormalize)

                    os.chdir("./TEST/")
                    imageOut("%04d" % i, combined_tensor[0], output_denormalize[0])
                    os.chdir("../")
                    i += 1


# test01()
test02()
