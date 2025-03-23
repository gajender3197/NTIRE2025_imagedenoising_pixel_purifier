import numpy as np
import imageio,os


def add_noise(image, sigma=50):
    """
    image: input image, numpy array, dtype=uint8, range=[0, 255]
    sigma: default 50
    """
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(0, sigma / 255, image.shape)
    gauss_noise = image + noise
    return gauss_noise * 255


def save_image(image, path):
    """
    image: saved image, numpy array, dtype=float
    path: saving path
    """
    # The type of the image is float, and range of the image might not be in [0, 255]
    # Thus, before saving the image, the image needs to be clipped.
    image = np.round(np.clip(image, 0, 255)).astype(np.uint8)
    imageio.imwrite(path, image)


def crop_image(image, s=8):
    h, w, c = image.shape
    image = image[:h - h % s, :w - w % s, :]
    return image


if __name__ == "__main__":
    input_dir = "/home/test/Deepak/NTIRE_2025_Denoising/NTIRE2025_Dn50_challenge-main/input/LSDIR_valid_HR"
    out_dir = "/home/test/Deepak/NTIRE_2025_Denoising/NTIRE2025_Dn50_challenge-main/input/LSDIR_valid_noise50"
    
    file_list = os.listdir(input_dir)
    for file in file_list:
        filename,ext = os.path.splitext(file)
        image_name = os.path.join(input_dir,file)
        #image_name = "example.png"

        img = imageio.imread(image_name)

        # The provided noisy validation images in the following link are cropped such that the width and height are multiples of 8.
        # https://drive.google.com/file/d/1iYurwSVBUxoN6fQwUGP-UbZkTZkippGx/view?usp=share_link
        # You can achieve that using the function crop_image.
        img = crop_image(img)

        # This image is used as the noisy image for training.
        img_noise = add_noise(img, sigma=50)

        # This function ensures that the image is properly clipped and rounded before saving.
        out_path = os.path.join(out_dir,filename+ext)
        save_image(img_noise, out_path)
