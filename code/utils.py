
from PIL import Image
import io
import base64
from pathlib import Path
import pandas as pd
import numpy as np
def resize_image_with_padding_png(input_path):
    desired_size = 200
    # Open the image
    img = Image.open(input_path)
    
    # Calculate the ratio to maintain aspect ratio
    ratio = max(desired_size / img.size[0], desired_size / img.size[1])
    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
    
    # Resize the image
    img_resized = img.resize(new_size, Image.ANTIALIAS)
    
    # Create a new white image
    new_img = Image.new("RGB", (desired_size, desired_size), color=(255, 255, 255))
    
    # Paste the resized image onto the center of the white image
    new_img.paste(img_resized, ((desired_size - new_size[0]) // 2,
                                (desired_size - new_size[1]) // 2))
    
    # Convert to bytes for HTML embedding
    img_byte_arr = io.BytesIO()
    new_img.save(img_byte_arr, format='PNG')  # Save as PNG
    img_byte_arr = img_byte_arr.getvalue()
    
    return img_byte_arr

def get_image_path(row):
    img_name = row["img_path"]
    path = Path(f"../data/q-{row['labeled_quality']}/{row['shape']}/{img_name}")
    return path.absolute()

def get_image_html(row):
    input_path = get_image_path(row)
    image_byte = resize_image_with_padding_png(input_path)
    base64_img = base64.b64encode(image_byte).decode('utf-8')
    img_html = f'<img src="data:image/png;base64,{base64_img}" />'
    return img_html


def percentile_color(data, column, reverse=False):
    """Function to calculate percentile and assign color based on ranking"""
    # Calculate percentile ranks; 100 is best, 0 is worst
    percentiles = data[column].rank(pct=True) * 100
    
    # If higher scores are better, and we want to reverse the colors for lower is better
    if reverse:
        percentiles = 100 - percentiles

    # Assign colors based on percentiles
    colors = percentiles.apply(lambda x: 
        'blue' if x > 66 else 
        'red' if x < 33 else 
        'yellow')
    
    return colors



def calculate_correctness(iqa_df, output_path=""):
    """Calculate the correctness of each metric and creates a html visualization
      of the metrics, if a path is provided."""

    # sort iqa_df by labeled_quality. Low first, then mid, then high
    quality_column = "labeled_quality" if "labeled_quality" in iqa_df.columns else "Quality"
    low = iqa_df[iqa_df[quality_column] == "low"]
    mid = iqa_df[iqa_df[quality_column] == "mid"]
    high = iqa_df[iqa_df[quality_column] == "high"]

    iqa_df = pd.concat([low, mid, high])
    scores_names = iqa_df.columns[3:]
    correctness_df =  iqa_df.iloc[:, 2:].copy()

    # Generate HTML table with dynamic color coding
    html = f'<table border="1"><tr><th>Image</th><th>Shape</th><th>Labeled Quality</th>'
    html += ''.join([f'<th>{col.upper()}</th>' for col in scores_names])
    html += '</tr>'

    for index, row in iqa_df.iterrows():
        html += f'<tr><td>{get_image_html(row)}</td><td>{row["shape"]}</td>'
        if row[quality_column] == "high":
            label_color = "blue"
        elif row[quality_column] == "mid":
            label_color = "yellow"
        else:
            label_color = "red"
        html += f'<td style="background-color: {label_color};">{row[quality_column]}</td>'
        for col in scores_names:
            if '↓' in col:
                column_colors = percentile_color(iqa_df, col, reverse=True)
            else:
                column_colors = percentile_color(iqa_df, col)
            color = column_colors[index]
            correct = 1 if color == label_color else 0
            correctness_df.loc[index, col] = correct
            html += f'<td style="background-color: {color};">{row[col]}</td>'
        html += '</tr>'
    html += '</table>'

    # Save the HTML content to a file
    if output_path:
        with open(output_path, 'w') as f:
            f.write(html)
    return correctness_df


    