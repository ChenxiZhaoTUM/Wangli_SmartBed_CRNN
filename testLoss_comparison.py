import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def loss_comparison():
    x_values = [25, 50, 85]
    y1_values = [139.158498, 132.712227, 126.727281]
    y2_values = [0.003014, 0.002837, 0.002747]
    y3_values = [10.851238, 10.213502, 9.888277]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 12))

    axes[0].plot(x_values, y1_values, marker='o', linestyle='-', color='b', label='Loss percentage of p')
    axes[0].set_xlabel('Train files')
    axes[0].set_ylabel('Y-axis 1')
    axes[0].legend()

    axes[1].plot(x_values, y2_values, marker='o', linestyle='-', color='r', label='MSE error')
    axes[1].set_xlabel('Train files')
    axes[1].set_ylabel('Y-axis 2')
    axes[1].legend()

    axes[2].plot(x_values, y3_values, marker='o', linestyle='-', color='y', label='Denormalized MSE error')
    axes[2].set_xlabel('Train files')
    axes[2].set_ylabel('Y-axis 3')
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def test_pic(pic_num):
    folder_path_1 = "TEST_CRNN_expo4_25files_01"
    folder_path_2 = "TEST_CRNN_expo4_50files_01"
    folder_path_3 = "TEST_CRNN_expo4_02"

    img_path_1 = os.path.join(folder_path_1, pic_num)
    img_path_2 = os.path.join(folder_path_2, pic_num)
    img_path_3 = os.path.join(folder_path_3, pic_num)

    img_1 = mpimg.imread(img_path_1)
    img_2 = mpimg.imread(img_path_2)
    img_3 = mpimg.imread(img_path_3)

    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    axes[0].imshow(img_1)
    axes[0].set_title('Image from Folder 25files')

    axes[1].imshow(img_2)
    axes[1].set_title('Image from Folder 50files')

    axes[2].imshow(img_3)
    axes[2].set_title('Image from Folder 85files')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


loss_comparison()
test_pic("0652.png")
test_pic("1364.png")
test_pic("2141.png")
test_pic("6900.png")
test_pic("7527.png")
