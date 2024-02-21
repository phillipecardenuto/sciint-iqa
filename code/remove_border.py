
"""
Remove panel border from extracted western blot

Author: João Phillipe Cardenuto (phillipe.cardenuto@ic.unicamp.br)
Recod.ai and Loyola University Chicago
Date: 2024-02-19
"""

import argparse
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from glob import glob
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

def check_black_border_thickness(img, position="top", threshold=10):
    if position == "top":
        for i in range(img.shape[0]-1):
            if img[i, img.shape[1]//2] > threshold:
                return i
    elif position == "bottom":
        for i in range(img.shape[0]-1, 0, -1):
            if img[i, img.shape[1]//2] > threshold:
                return i
    elif position == "left":
        for i in range(img.shape[1]-1):
            if img[img.shape[0]//2, i] > threshold:
                return i
    elif position == "right":
        for i in range(img.shape[1]-1, 0, -1):
            if img[img.shape[0]//2, i] > threshold:
                return i
    return 0

def check_white_border_thickness(img, position="top", threshold=250):
    if position == "top":
        for i in range(img.shape[0]-1):
            if img[i, img.shape[1]//2] < threshold:
                return i
    elif position == "bottom":
        for i in range(img.shape[0]-1, 0, -1):
            if img[i, img.shape[1]//2] < threshold:
                return i
    elif position == "left":
        for i in range(img.shape[1]-1):
            if img[img.shape[0]//2, i] < threshold:
                return i
    elif position == "right":
        for i in range(img.shape[1]-1, 0, -1):
            if img[img.shape[0]//2, i] < threshold:
                return i
    return 0

def remove_black_boder(img, threshold):
    # check the thickness of the border from top->bottom and left->right and
    # from bottom->top and right->left
    img = img.crop((3, 3, img.width-3, img.height-3))
    tmp_img = img.copy().convert("L")
    tmp_img = np.array(tmp_img)
    # top->bottom
    top = check_black_border_thickness(tmp_img, position="top", threshold=threshold)
    # bottom->top
    bottom = check_black_border_thickness(tmp_img, position="bottom", threshold=threshold)
    # left->right
    left = check_black_border_thickness(tmp_img, position="left", threshold=threshold)
    # right->left
    right = check_black_border_thickness(tmp_img, position="right", threshold=threshold)

    # return img[top:bottom, left:right]
    if right - left < 64 or bottom - top < 64:
        return img
    return img.crop((left, top, right, bottom))
    # return  left +3, top +3, right -3, bottom -3

def remove_white_boder(img, threshold):
    # check the thickness of the border from top->bottom and left->right and
    # from bottom->top and right->left
    tmp_img = img.copy().convert("L")
    tmp_img = np.array(tmp_img)
    # top->bottom
    top = check_white_border_thickness(tmp_img, position="top", threshold=threshold)
    # bottom->top
    bottom = check_white_border_thickness(tmp_img, position="bottom", threshold=threshold)
    # left->right
    left = check_white_border_thickness(tmp_img, position="left", threshold=threshold)
    # right->left
    right = check_white_border_thickness(tmp_img, position="right", threshold=threshold)

    # return img[top:bottom, left:right]
    if right - left < 64 or bottom - top < 64:
        return img
    return  img.crop((left, top, right, bottom))



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', '-i',required=True, type=str, help='path/to/figure. You can input multiple figures at the same time')
    parser.add_argument('--output-path','-o', required=True, type=str, help='Directory path where the extracted figures will be saved')
    opt = parser.parse_args()
    return opt

def process_image(input):
    img_path, output_path = input
    img_path = Path(img_path)
    output_path = Path(output_path)
    img = Image.open(img_path)
    img = remove_black_boder(img, threshold=10)
    img = remove_white_boder(img, threshold=250)
    if img.size[0] < 64 or img.size[1] < 64:
        return
    img.save(output_path.joinpath(img_path.name))

def run(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    # for img_path in tqdm(glob(input_path.as_posix() + "/*")):
    images = glob(input_path.as_posix() + "/*Blots.png")
    images.sort()
    input = [(img_path, output_path) for img_path in images]
    process_map(process_image,
                input, 
                max_workers=30, chunksize=1)                


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))


