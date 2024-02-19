"""
Process all iqa metrics in parallel for the dataset.

Author: João Phillipe Cardenuto (phillipe.cardenuto@ic.unicamp.br)
Recod.ai and Loyola University Chicago
Date: 2024-02-19
"""

import numpy as np
import pyiqa
from tqdm import tqdm
from PIL import Image
import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from glob import glob
from metrics import estimate_compression_level, count_uniform_squares, get_iqa_score


def process_image(image_path):
        # Get image name
        img_name = Path(image_path).name
        # Get image quality (low, mid, high)
        quality = image_path.split("/")[1].replace("q-", "")
        # Get image shape type (super-wide, wide, tall, square)
        shape = Path(image_path).parent.name
        # Estimate compression level
        compression = estimate_compression_level(image_path)
        # Estimate pixelation
        pixelation = count_uniform_squares(image_path)[2]

        row = [img_name, shape, quality, compression, pixelation]

        # Get IQA scores
        for model in pyiqa.list_models("NR"):
            score = get_iqa_score(image_path, model, "NR")
            row.append(score)
        return row

# Load the image paths
images = glob("../data/**/*.png", recursive=True)

# Create a dataframe to store the results
iqa_df = pd.DataFrame(columns=["img_path", "shape", "labeled_quality", "pred_compression", "pred_pixelation"] + pyiqa.list_models("NR"))

# Create a lock to avoid losing data while updating the dataframe
df_lock = Lock()

# Process the images in parallel
with ThreadPoolExecutor(max_workers=20) as executor:
    future_to_image = [executor.submit(process_image, image_path) for image_path in images]
    with tqdm(total=len(future_to_image), desc="Processing") as pbar:
        for future in as_completed(future_to_image):
            row = future.result()
            with df_lock:
                iqa_df = pd.concat([iqa_df, pd.DataFrame([row], columns=iqa_df.columns)], ignore_index=True)
            pbar.update(1)


iqa_df.to_csv("iqa_results.csv", index=False)

# Rename columns to include the direction of the metric 
# "↑" means that higher values are better
# "↓" means that lower values are better
for m in pyiqa.list_models("NR"):
    s_better = "↓"  if pyiqa.create_metric(m).lower_better else "↑"
    if m in iqa_df.columns:
        renamed_c = f"{m} {s_better}"
        iqa_df = iqa_df.rename(columns={m: renamed_c})

m = 'pred_compression'
renamed_c = f"{m} ↓"
iqa_df = iqa_df.rename(columns={m: renamed_c})
m = 'pred_pixelation'
renamed_c = f"{m} ↓"
iqa_df = iqa_df.rename(columns={m: renamed_c})

iqa_df.to_csv("iqa_results_with_arrows.csv", index=False)
