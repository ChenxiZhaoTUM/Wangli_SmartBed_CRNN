import math
import os
from tkinter import Tk, Label, PhotoImage
from PIL import Image, ImageTk
from matplotlib import pyplot as plt


class ImageDisplay:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_list = self.get_png_images()
        self.current_index = 0

        self.root = Tk()
        self.root.title("Image Display")
        self.label = Label(self.root)
        self.label.pack()

        self.display_image()
        self.root.after(500, self.next_image_auto)
        self.root.mainloop()

    def get_png_images(self):
        png_images = [f for f in os.listdir(self.folder_path) if f.lower().endswith(".png")]
        return png_images

    def display_image(self):
        if self.image_list:
            image_path = os.path.join(self.folder_path, self.image_list[self.current_index])
            image = Image.open(image_path)
            image = image.resize((800, 800), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)

            self.label.config(image=photo)
            self.label.image = photo

    def next_image_auto(self):
        self.current_index = (self.current_index + 1) % len(self.image_list)
        self.display_image()
        self.root.after(500, self.next_image_auto)


def LossDisplay(train_file_path, vali_file_path, filter_threshold=None):
    with open(train_file_path, 'r') as file:
        train_loss = [float(line.strip()) for line in file.readlines()]

    with open(vali_file_path, 'r') as file:
        vali_loss = [float(line.strip()) for line in file.readlines()]

    iteration_numbers_train = list(range(len(train_loss)))
    iteration_numbers_vali = list(range(len(vali_loss)))

    if filter_threshold is not None:
        filtered_data_train = [(iteration, loss) for iteration, loss in zip(iteration_numbers_train, train_loss) if
                               loss <= filter_threshold]
        iteration_numbers_train, train_loss = zip(*filtered_data_train)

        filtered_data_vali = [(iteration, loss) for iteration, loss in zip(iteration_numbers_vali, vali_loss) if
                              loss <= filter_threshold]
        iteration_numbers_vali, vali_loss = zip(*filtered_data_vali)

    plt.figure(figsize=(14, 12))

    plt.subplot(2, 1, 1)
    plt.plot(iteration_numbers_train, train_loss, label='Train Loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('loss of train')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, len(iteration_numbers_train))
    plt.ylim(min(train_loss), max(train_loss))

    plt.subplot(2, 1, 2)
    plt.plot(iteration_numbers_vali, vali_loss, color='orange', label='Validation Loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('loss of validation')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, len(iteration_numbers_vali))
    plt.ylim(min(vali_loss), max(vali_loss))

    plt.tight_layout()
    plt.show()


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


def train(epochs, minLR, maxLR):
    learning_rates = []
    for i in range(epochs):
        lr = computeLR(i, epochs, minLR, maxLR)
        learning_rates.append(lr)
    return learning_rates


def plot_lr(epoch_lr):
    plt.plot(range(1, len(epoch_lr) + 1), epoch_lr)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Epoch vs Learning Rate')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # folder_path = "./TEST_UNet3D_02"
    # ImageDisplay(folder_path)

    # LossDisplay("./UNet3D_03_MSELoss.txt", "./UNet3D_03_MSELossVal.txt", filter_threshold=0.005)

    lrG = 0.0006
    epochs = 1000000
    learning_rates = train(epochs, lrG * 0.1, lrG)
    plot_lr(learning_rates)
