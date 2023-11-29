import os
from tkinter import Tk, Label, PhotoImage
from PIL import Image, ImageTk


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


if __name__ == "__main__":
    folder_path = "./TEST_UNet3D_02"
    ImageDisplay(folder_path)
