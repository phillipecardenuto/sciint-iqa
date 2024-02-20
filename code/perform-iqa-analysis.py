"""
This file perform an iqa-analysis based on the best-effort techniques for each metric. The metrics are:
- Compression: Estimate the compression level of an image based on the number of zeros in the DCT coefficients.
- Pixelation: Count 2x2 squares with all the same values in an image.
- ManiQA-PIPAl: ManiQA-PIPAl is a no-reference image quality assessment metric that uses a pre-trained model to estimate the quality of an image.
- NIQE: Natural Image Quality Evaluator (NIQE) is a no-reference image quality assessment metric that uses a pre-trained model to estimate the quality of an image.

Author: João Phillipe Cardenuto (phillipe.cardenuto@ic.unicamp.br)
Recod.ai and Loyola University Chicago
Date: 2024-02-19
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from scipy.fftpack import dct  # Import DCT from scipy
from tqdm.contrib.concurrent import process_map
import os
import pandas as pd
import torch
from pyiqa import create_metric
from PIL import Image
from tqdm import tqdm

# Set torch cache dir


def estimate_compression_level(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    blocks = [img[i:i+8, j:j+8] for i in range(0, h, 8) for j in range(0, w, 8)]
    dct_zeros = 0
    for block in blocks:
        block_dct = dct(dct(block.T, norm='ortho').T, norm='ortho')
        dct_zeros += np.sum(block_dct == 0)
    compression_estimate = dct_zeros / len(blocks)
    return compression_estimate

def count_uniform_squares(image_path):
    img = cv2.imread(image_path)
    max_size = 1024
    scale = max_size / max(img.shape[:2])
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    h, w = img.shape[:2]
    v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))[2]
    diff_horizontal = np.abs(np.diff(v, axis=1))[1:h, :w]
    diff_vertical = np.abs(np.diff(v, axis=0))[:h, 1:w]
    diff_squares = diff_horizontal[:-1, :-1] + diff_horizontal[1:, :-1] + diff_vertical[:-1, :-1] + diff_vertical[:-1, 1:]
    uniform_count = np.sum(diff_squares == 0)
    total_squares = (h - 1) * (w - 1)
    fraction_uniform = uniform_count / total_squares
    return uniform_count, total_squares, fraction_uniform

def get_iqa_score(image_path):
    try:
        # img = Image.open(image_path).convert('RGB')
        # img = np.array(img).astype(np.float32) / 255.0
        # img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        ref_img_path = None
        score = iqa_model(image_path, ref_img_path).cpu().item()
    except Exception as e:
        print(e)
        score = None
    return score

def process_image(input):
    image_path, metric_name, device = input
    if metric_name.lower() in ["maniqa-pipal", 'tres-flive', 'pi',  "clipiqa", 'niqe']:
        score = get_iqa_score(image_path)
    elif metric_name.lower() == "compression":
        score = estimate_compression_level(image_path)
    elif metric_name.lower() == "pixelation":
        score = count_uniform_squares(image_path)[2]
    return [image_path, score]

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', '-i', required=True, type=str, help='Path to input images')
    parser.add_argument('--output-path', '-o', required=True, type=str, help='Path for the output CSV file')
    parser.add_argument('--metric-name', '-m', required=True, type=str, help='Metric name for image assessment')
    parser.add_argument('--device', '-d', default=0, type=int, help='GPU device ID')
    return parser.parse_args()

def run(input_path, output_path, metric_name, device):
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if os.path.isfile(output_path.joinpath(f"{metric_name}.csv")):
        processed_df = pd.read_csv(output_path.joinpath(f"{metric_name}.csv"))
        processed = processed_df["image_path"].values.tolist()
    else:
        processed = []
    images = list(input_path.glob("*Blots.png"))
    images.sort()
    images = [img for img in tqdm(images, desc="Analyzing Processed Images") if img.as_posix() not in processed]
    if len(images) == 0:
        print(f"No new images to process")
        return
    
    inputs = [(str(img_path), metric_name, device) for img_path in images]
    if metric_name.lower() in ["maniqa-pipal", 'niqe']:
        results = pd.DataFrame([process_image(i) for i in tqdm(inputs)], columns=["image_path", metric_name])
    else:
        results = pd.DataFrame(process_map(process_image, inputs, max_workers=30, chunksize=1), columns=["image_path", metric_name])
    
    if processed:
        results = pd.concat([processed_df, results], ignore_index=True)
    results.to_csv(output_path.joinpath(f"{metric_name}.csv"), index=False)

if __name__ == "__main__":
    opt = parse_opt()
    device_str = f"cuda:{opt.device}" if torch.cuda.is_available() and opt.device >= 0 else "cpu"
    device = torch.device(device_str)
    print(f"Using {device_str}")
    if not opt.metric_name.lower() in ["compression", "pixelation", "maniqa-pipal", 'niqe']:
        raise ValueError("Invalid metric name")
    if opt.metric_name.lower() in ["maniqa-pipal", 'niqe']:
        iqa_model = create_metric(opt.metric_name.lower(), metric_mode="NR", device=device)
    else:
        iqa_model = None
    run(opt.input_path, opt.output_path, opt.metric_name, device)

