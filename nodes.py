from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms as T

class ImageToAscii:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "gen_ascii"

    CATEGORY = "conditioning"
    ASCII_CHARS = "@%#*+=-:. "
 



    def resize_image(self,image, new_width=100):
        width, height = image.size
        aspect_ratio = height / float(width)
        new_height = int(aspect_ratio * new_width * 0.55)
        resized_image = image.resize((new_width, new_height))
        return resized_image

    def grayscale_image(self,image):
        return image.convert("L")

    def pixel_to_ascii(self,image):
        pixels = np.array(image)
        ascii_str = ""
        for row in pixels:
            for pixel in row:
                ascii_str += self.ASCII_CHARS[pixel // 32]
            ascii_str += "\n"
        return ascii_str

    def gen_ascii(self,image, new_width=64):
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image.squeeze(0)
            image = image.cpu().numpy()
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image, mode='RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: Expected a torch.Tensor, PIL.Image, or NumPy array, but got {type(image)}")
        #image = T.ToPILImage()(image)
        image = self.resize_image(image, new_width)
        image = self.grayscale_image(image)

        ascii_art = self.pixel_to_ascii(image)
        with open("ascii_image.txt", "w") as f:
            f.write(ascii_art)

        print(ascii_art)
        return (ascii_art,)