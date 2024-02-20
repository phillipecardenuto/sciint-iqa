import argparse
from pathlib import Path
import numpy as np
import os
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import easyocr

# Set torch cache dir

def detect_text_in_image(image_path, reader):
    """
    Detects text in a single image using EasyOCR.

    Args:
        image_path (str): Path to the image file.
        reader (easyocr.Reader): An EasyOCR Reader object configured for the desired language.

    Returns:
        bool: True if text is detected, False otherwise.
    """
    try:
        results = reader.readtext(image_path, detail=1)
        if results:  # If the list is not empty, text was detected
            text = [r[-1] for r in results]
            max_confidence = max(text)
            return max_confidence
        else:
            return 0
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return 0


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', '-s', required=True, type=str, help='Work on a split of the dataset. For instance [0-10] will work on'\
                                                        'the first 10\% of the dataset')
    return parser.parse_args()

def run():
    df = pd.read_csv("merged.csv")
    start, end = [int(x) for x in opt.split[1:-1].split("-")]
    start = int(len(df) * start / 100)
    end = int(len(df) * end / 100)
    processing_df = df.iloc[start:end]
        
    if len(df) == 0:
        print(f"No data to process")
        return
    
    for i, row in tqdm(processing_df.iterrows(), total=len(processing_df), desc="Processing images"):
        image_path = row["image_path"]
        if not Path(image_path).exists():
            print(f"Image {image_path} does not exist")
            continue

        processing_df.loc[i, "has_text"] = detect_text_in_image(image_path, reader)

    processing_df.to_csv(f"split_{start}_{end}.csv", index=False)

if __name__ == "__main__":
    opt = parse_opt()
    reader = easyocr.Reader(['en'], gpu=True)
    run()

