import base64
import requests
local_img_path = './test.jpg'
with open(local_img_path, 'rb') as f:
    base64_bytes = base64.encodebytes(f.read())
    # or
    # base64.encodebytes(f.read())
api_url = 'http://localhost:5000/api/ocr_dec'
# or
# api_url = 'http://0.0.0.0:5000/api/ocr_dec'
data = {
    'img_base64': base64_bytes
    # or
    # 'img_base64': base64_bytes.decode()
}
response = requests.post(api_url, data=data)
json = response.json()
print(json)
