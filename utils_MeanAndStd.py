import math, os
import numpy as np
import matplotlib.pyplot as plt
import torch

#NOTE: the range of the lambda output should be [0,1]                   
def get_cosine_lambda(initial_lr,final_lr,epochs,warmup_epoch):
    """
    Returns a lambda function that calculates the learning rate based on the cosine schedule.

    Args:
        initial_lr (float): The initial learning rate.
        final_lr (float): The final learning rate.
        epochs (int): The total number of epochs.
        warmup_epoch (int): The number of warm-up epochs.

    Returns:
        function: The lambda function that calculates the learning rate.
    """
    def cosine_lambda(idx_epoch):
        if idx_epoch < warmup_epoch:
            return idx_epoch / warmup_epoch
        else:
            return 1-(1-(math.cos((idx_epoch-warmup_epoch)/(epochs-warmup_epoch)*math.pi)+1)/2)*(1-final_lr/initial_lr)
    return cosine_lambda

# compute learning rate with decay in second half
def computeLR(i, epochs, minLR, maxLR):
    if i < epochs * 0.5:
        return maxLR
    e = (i / float(epochs) - 0.5) * 2.
    # rescale second half to min/max range
    fmin = 0.
    fmax = 6.
    e = fmin + e * (fmax - fmin)
    f = math.pow(0.5, e)
    return minLR + (maxLR - minLR) * f


def makeDirs(directoryList):
    for directory in directoryList:
        if not os.path.exists(directory):
            os.makedirs(directory)


input_mean = torch.load('dataset/saved_tensor_for_train/input_mean.pt')
input_std = torch.load('dataset/saved_tensor_for_train/input_std.pt')

def imageOut(filename, _input, _target, _output, max_val=40, min_val=0):
    target = np.copy(_target)
    output = np.copy(_output)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

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
    target_image = np.reshape(target, (32, 64))
    im2 = ax2.imshow(target_image, cmap='jet', vmin=min_val, vmax=max_val)
    ax2.axis('off')
    cbar2 = fig.colorbar(im2, ax=ax2)

    ax3.set_aspect('equal', 'box')
    output_image = np.reshape(output, (32, 64))
    im3 = ax3.imshow(output_image, cmap='jet', vmin=min_val, vmax=max_val)
    ax3.axis('off')
    cbar3 = fig.colorbar(im3, ax=ax3)

    plt.tight_layout()
    save_path = os.path.join(filename)
    plt.savefig(save_path)
    plt.close(fig)


def log(file, line, doPrint=True):
    f = open(file, "a+")
    f.write(line + "\n")
    f.close()
    if doPrint:
        print(line)


# reset log file
def resetLog(file):
    f = open(file, "w")
    f.close()
