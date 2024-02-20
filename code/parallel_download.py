"""This code was provided by UNINA for downloading figures from Figshare. 
It was improved by UNICAMP to download the figures in parallel, discarding small or unknow figure formats.
It also creates a metadata file with information about the downloaded figures.

Date: 2024-01-06
"""

import requests
from tqdm import tqdm
import os
import csv
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from threading import Lock
import io
from time import sleep

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--search', type=str)
args = parser.parse_args()

search_for = args.search
output_folder = <output-folder>
meta_file = <meta-file-name> 
search_url = 'https://api.figshare.com/v2/articles/search'

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Initialize meta file
if not os.path.isfile(meta_file):
    with open(meta_file, 'w') as meta:
        writer = csv.writer(meta)
        writer.writerow(['file_id', 'article_id', 'license', 'file_url', 'article_url', 'original_filename', 'search_string', 'md5', 'status'])

def fetch_and_check_image(file_info):
    file_id, article_id, license, download_url, article_url, filename, search_for, md5 = file_info
    ext = filename.split('.')[-1].lower()
    output = os.path.join(output_folder, f"{article_id}_{file_id}.{ext}")
    
    # Check if file already exists to avoid re-downloading
    if not os.path.isfile(output):
        try:
            response = requests.get(download_url)
            if response.status_code == 200:
                # Check if the content is TIFF or PNG
                if ext in ['tiff', 'tif', 'png', 'jpg', 'jpeg', "pdf"]:
                    if ext == 'pdf':
                        with open(output, 'wb') as f:
                            f.write(response.content)
                        return file_info + ('downloaded',)
                    image = Image.open(io.BytesIO(response.content))
                    width, height = image.size
                    # Check if the smallest dimension is greater than 1000px
                    if min(width, height) > 1000:
                        with open(output, 'wb') as f:
                            f.write(response.content)
                        return file_info + ('downloaded',)
                    else:
                        return file_info + ('skipped - size',)
                else:
                    return file_info + ('skipped - format',)
            else:
                return file_info + ('error - download failed',)
        except Exception as e:
            return file_info + ('error - exception',)
    return file_info + ('exists',)

def main():
    i = 1
    write_lock = Lock() 
    while True:

        try:
            print(f'Page {i}')
            myobj = {"search_for": search_for, "item_type": 1, "page": i, "page_size": 300}
            results = requests.post(search_url, json=myobj).json()
            
            if not results:  # Break the loop if there are no results
                break

            tasks = []
            for r in results:
                article_id = r['id']
                article = requests.get(f"https://api.figshare.com/v2/articles/{article_id}").json()
                if 'files' in article and not article.get('is_confidential', False):
                    license = article['license']['name']
                    for f in article['files']:
                        file_info = (
                            f['id'],
                            article_id,
                            license,
                            f['download_url'],
                            article['figshare_url'],
                            f['name'],
                            search_for,
                            f['supplied_md5']
                        )
                        tasks.append(file_info)
            
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(fetch_and_check_image, task) for task in tasks]
                with open(meta_file, 'a') as meta, tqdm(total=len(futures), desc="Downloading") as pbar:
                    writer = csv.writer(meta)
                    for future in as_completed(futures):
                        result = future.result()
                        # Acquire the lock before writing to the file
                        with write_lock:
                            writer.writerow(result)
                        # Automatically releases the lock at the end of the with block
                        pbar.update(1)
            
            i += 1
        except Exception as e:
            print(f"Error on page {i}: {e}")
            i+=1
            sleep(300)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
