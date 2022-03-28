import os
import urllib.request
import urllib.parse
import urllib
import time
import pandas as pd

df = pd.read_excel('/run/user/1000/gvfs/sftp:host=vickys-mac-mini.local/Users/vx/Documents/GitHub/PaddleOCR/xlsx/prices.xlsx')
row_num, col_num = df.shape[0], df.shape[1]

for id in range(row_num):
    # print(df['image_url'][id])
    if (str(id) + ".jpg") not in os.listdir():
        fullname = str(id) + ".jpg"
        # store image url
        url = df['image_url'][id]
        try:
            urllib.request.urlretrieve(url, fullname)
            print("scraped!")
        except ConnectionResetError:
            time.sleep(5)
            print("Sleep 5 secs and reset connection")
        except urllib.error.HTTPError as err:
            url = urllib.parse.quote(url, ':/=&?')
            print("HTTP Error")
            continue
        except urllib.error.URLError as err:
            print("URL Error")
            continue
    else:
        continue
