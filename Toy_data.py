import cv2
import os
import random
import numpy as np
from skimage.util import random_noise


def circle(color, if_noise=False):

    height = 64
    width = 64

    back_image = np.ones((height, width, 3), np.uint8)
    choice_back = ["black", "white"]
    back_color = random.choice(choice_back)

    if back_color == "black":
        back_pixel = (0, 0, 0)
    elif back_color == "white":
        back_pixel = (255, 255, 255)

    back_image[:, :] = back_pixel

    radius = random.randint(10, 30)
    center_h = random.randint(radius, 64-radius)
    center_w = random.randint(radius, 64-radius)
    center_coordinates = (center_h, center_w)

    image = cv2.circle(back_image, center_coordinates, radius, color, -1)
    
    if if_noise == True:
        sp_amount = random.random()
        sp_ratio = random.random()
        image = sp_noise(image, amount=sp_amount, salt_vs_pepper=sp_ratio)

    return image
    


def rectangle(color, if_noise=False):

    height = 64
    width = 64

    back_image = np.ones((height, width, 3), np.uint8)
    choice_back = ["black", "white"]
    back_color = random.choice(choice_back)

    if back_color == "black":
        back_pixel = (0, 0, 0)
    elif back_color == "white":
        back_pixel = (255, 255, 255)

    back_image[:, :] = back_pixel

    start_h = random.randint(10, 30)
    start_w = random.randint(10, 30)
    end_h = random.randint(start_h, 64-start_h)
    end_w = random.randint(start_w, 64-start_w)
    start = (start_h, start_w)
    end = (end_h, end_w)

    image = cv2.rectangle(back_image, start, end, color, -1)
    if if_noise == True:
        sp_amount = random.random()
        sp_ratio = random.random()
        image = sp_noise(image, amount=sp_amount, salt_vs_pepper=sp_ratio)

    return image


def ellipse(color, if_noise=False):

    height = 64
    width = 64

    back_image = np.ones((height, width, 3), np.uint8)
    choice_back = ["black", "white"]
    back_color = random.choice(choice_back)

    if back_color == "black":
        back_pixel = (0, 0, 0)
    elif back_color == "white":
        back_pixel = (255, 255, 255)

    back_image[:, :] = back_pixel

    a = random.randint(10, 30)
    b = random.randint(10, 30)
    center_h = random.randint(a, 64-a)
    center_w = random.randint(b, 64-b)
    angle = random.randint(0, 360)
    start_angle = random.randint(0, 360)
    end_angle = random.randint(0, 360)


def sp_noise(image, amount, salt_vs_pepper):

    white_num = int(64*64*amount*salt_vs_pepper)
    black_num = int(64*64*amount*(1-salt_vs_pepper))

    for w in range(white_num):

        y_coord=random.randint(0, 64 - 1)
        x_coord=random.randint(0, 64 - 1)
        # Color that pixel to white
        image[y_coord, x_coord, :] = (255, 255, 255)

    for b in range(black_num):

        y_coord=random.randint(0, 64 - 1)
        x_coord=random.randint(0, 64 - 1)
        # Color that pixel to white
        image[y_coord, x_coord, :] = (0, 0, 0)

    return image

 
if __name__ == "__main__":

    save_path = "D://projects//open_cross_entropy//code//toy_data_test_inliers//"
    num_imgs = 100
    shape = "circleGreen"
    color = (0, 255, 0)
    noising = False

    for i in range(num_imgs):
        
        if shape == "circle" or shape == "circleRed" or shape == "circleGreen":
            img = circle(color, noising)
        elif shape == "rectangle" or shape == "rectangleBlue" or shape == "rectangleGreen":
            img = rectangle(color, noising)

        img_name = shape + "_" + str(i) + ".png"
        cv2.imwrite(save_path + img_name, img)