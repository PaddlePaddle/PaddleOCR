from flask import request, Flask, request
from shutil import copy
import os
from typing import Dict, List, Union
from urllib.request import urlretrieve
from pathlib import Path
from paddleocr import PaddleOCR
from hashlib import md5
import time
import base64
ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False)
app = Flask(__name__)
img_dir = Path('imgs')
img_dir.mkdir(exist_ok=True)
ocr_cache_dict = dict()


def get_dict_from_request() -> dict:
    """
    get json data from request as much as possible

    Returns
    -------
    dict
        request data in dict format    
    """
    json = {**request.args}
    if request.json:
        json = {**json, **request.json}
    if request.form:
        json = {**json, **request.form.to_dict()}
    return json


def download_image(img_url: str) -> str:
    """
    download image or copy image to local from url 

    Parameters
    ----------
    img_url : str
        url of image to be downloaded

    Returns
    -------
    str
        local file path of image

    Notes
    -----
    if download failed, empty string `''` will be returned
    """
    d = md5(str(img_url).encode()).hexdigest()
    file_name = f'{img_dir}/{d}.jpg'
    # NOTE: insecurity
    # # copy from local file system in the running container
    # if Path(img_url).exists():
    #     copy(img_url, file_name)

    if Path(file_name).exists():
        return file_name
    # download from internet
    try:
        urlretrieve(img_url, file_name)
        return file_name
    except:
        return ''


def base64_to_file(s: Union[str, bytes]) -> str:
    """
    decode base64 string or bytes and save to local file system

    Parameters
    ----------
    s : Union[str, bytes]
        base64 string or bytes

    Returns
    -------
    str
        local file path of base64 data
    """
    d = md5(str(s).encode()).hexdigest()
    file_name = f'{img_dir}/{d}.jpg'
    if Path(file_name).exists():
        return file_name
    if isinstance(s, str):
        b = base64.decodebytes(s.encode())
    elif isinstance(s, bytes):
        b = base64.decodebytes(s)
    else:
        return ''
    with open(file_name, 'wb') as f:
        f.write(b)
    return file_name


@app.route('/api/ocr_dec', methods=['POST'])
def ocr_text() -> None:
    """
    ocr web api that accept image url, image path and base64 data of image 
    """
    st = time.time()
    json = get_dict_from_request()
    img_url: str = json.get('img_url')
    base64_data: str = json.get('img_base64')
    img_path = ''
    if img_url:
        img_path = download_image(img_url)
    elif base64_data:
        img_path = base64_to_file(base64_data)
    if not img_path:
        et = time.time()
        return {
            'success': False,
            'time_cost': et-st,
            'results': [],
            'msg': 'maybe img_url or img_base64 is wrong'
        }
    results = ocr_cache_dict.get(img_path)
    if not results:
        ocr_result_list = ocr.ocr(img_path)

        et = time.time()
        if ocr_result_list is None:
            ocr_result_list = []
            os.remove(img_path)
        else:
            # make sure float32 can be JSON serializable
            ocr_result_list: list = eval(str(ocr_result_list))
            results: List[Dict] = []
            for each in ocr_result_list:
                item = {
                    'confidence': each[-1][1],
                    'text': each[-1][0],
                    'text_region': each[:-1]
                }
                results.append(item)
            ocr_cache_dict[img_path] = results

    et = time.time()
    return {
        'success': True,
        'time_cost': et-st,
        'results': results,
        'msg': ''
    }


if __name__ == '__main__':
    port = os.environ.get('FLASK_PORT', '')
    if port.isalnum() and int(port) > 0:
        port = int(port)
    else:
        port = 5000
    app.run(host='0.0.0.0', port=port)
