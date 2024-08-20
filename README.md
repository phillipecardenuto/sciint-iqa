# Raw Western Blot Image Quality Assessment

This repository provides an image quality assessment (IQA) of raw western blot figures downloaded from Figshare. Through our analysis, we observed that many images labeled as raw data (unedited or unprocessed) still contain artifacts and noise, leading to a reduction in overall image quality.

## Dataset

We analyzed a dataset of approximately 95,000 electrophoresis images (gel blots) from 15,011 different research publications. Our findings indicate that over 85% of these images can be classified as low-quality based on various IQA metrics.

## Tools and Methodology

For the IQA, we utilized the [PyIQA toolbox](https://github.com/chaofengc/IQA-PyTorch), which implements a variety of image quality metrics.

The metric scores for each image are available in the `figshare-blots.csv` file located in the `data` directory. This file also includes metadata required to download the images from Figshare.

The complete analysis can be found in the `perform-iqa-analysis.ipynb` notebook, which is located in the `notebooks` directory.

## Image Quality Results

The table below summarizes the percentage of images that meet different quality thresholds based on individual IQA metrics:

| Metric           | Mild Quality (%) | OK Quality (%) | High Quality (%) |
|------------------|------------------|----------------|------------------|
| maniqa-pipal     | 36.21            | 8.63           | 8.40             |
| niqe             | 34.27            | 33.46          | 9.39             |
| pixelation       | 9.47             | 3.56           | 2.40             |
| compression      | 54.88            | 33.56          | 18.37            |

### Combined Metrics

When considering all IQA metrics simultaneously, the percentage of images that meet the quality thresholds is as follows:

| Mild Quality (%) | OK Quality (%) | High Quality (%) |
|------------------|----------------|------------------|
| 12.48            | 3.24           | 1.10             |

---
