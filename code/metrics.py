"""
This file contains the implementation of the metrics used to evaluate the quality of the images.

Author: João Phillipe Cardenuto (phillipe.cardenuto@ic.unicamp.br)
Recod.ai and Loyola University Chicago
Date: 2024-02-19
"""


from scipy.fftpack import dct
import numpy as np
import cv2
from pyiqa import create_metric
import pyiqa
from tqdm import tqdm
from PIL import Image
import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from glob import glob

def estimate_compression_level(image_path):
    """
    Estimate the JPEG compression level of an image, based on the number of zeros in the DCT coefficients.
    """
    # Load the image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    
    # Break the image into 8x8 blocks
    blocks = [img[i:i+8, j:j+8] for i in range(0, h, 8) for j in range(0, w, 8)]
    
    # Analyze DCT coefficients
    dct_zeros = 0
    for block in blocks:
        # Apply DCTb
        block_dct = dct(dct(block.T, norm='ortho').T, norm='ortho')
        # Count zeros (quantized coefficients)
        dct_zeros += np.sum(block_dct == 0)
    
    # Heuristic to estimate compression based on zeros in DCT coefficients
    compression_estimate = dct_zeros / len(blocks)
    
    return compression_estimate

def count_uniform_squares(image_path):
    """
    Count 2x2 squares with all the same values in an image.

    Args:
    - image_path: Path to the image file.

    Returns:
    - count: Number of uniform 2x2 squares.
    - total: Total number of 2x2 squares in the image.
    - fraction: Fraction of 2x2 squares that are uniform.
    """
    # Load image in grayscale to simplify analysis
    img = cv2.imread(image_path)

    max_size = 1024
    if img.shape[0] > img.shape[1]:
        scale = max_size / img.shape[0]
    else:
        scale = max_size / img.shape[1]
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    h,s,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    
    h, w = img.shape[:2	]
    # Calculate differences between adjacent pixels horizontally and vertically
    diff_horizontal = np.abs(np.diff(v, axis=1))[1:h, :w]
    diff_vertical = np.abs(np.diff(v, axis=0))[:h, 1:w]
    
    # Calculate the difference for 2x2 squares (bottom right pixel compared to others)
    diff_squares = diff_horizontal[:-1, :-1] + diff_horizontal[1:, :-1] + diff_vertical[:-1, :-1] + diff_vertical[:-1, 1:]
    
    # Count squares where all differences are zero (indicating uniformity)
    uniform_count = np.sum(diff_squares == 0)
    
    # Total number of 2x2 squares
    total_squares = (img.shape[0] - 1) * (img.shape[1] - 1)
    
    # Fraction of uniform 2x2 squares
    fraction_uniform = uniform_count / total_squares
    
    return uniform_count, total_squares, fraction_uniform


def get_iqa_score(input, metric_name, metric_mode, ref_img_path=None):
    """
    Get the score of an image using a given IQA metric, implemented in the pyiqa package.

    Args:
        - input: Image to be evaluated.
        - metric_name: Name of the IQA metric to be used.
        - metric_mode: NR or FR, indicating whether the metric is full-reference or no-reference.
        - ref_img_path: Path to the reference image, in case of full-reference metrics.

    Returns:
        - score: Quality score of the image.
    
    Exceptions will be ignores and the function will return None in case of error.
    """
    try:
        metric_name = metric_name.lower()
        iqa_model = create_metric(metric_name, metric_mode=metric_mode)
        metric_mode = iqa_model.metric_mode
        if metric_mode == pyiqa.MetricMode.NR:
            ref_img_path = None
        score = iqa_model(input, ref_img_path).cpu().item()
    except Exception as e:
        print(e)
        score = None

    return score
