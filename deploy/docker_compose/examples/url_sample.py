import requests
remote_img_url = 'https://ai.bdstatic.com/file/5419067D0B374C12A8CFB5C74684CC06'
data = {
    'img_url': remote_img_url
}
api_url = 'http://localhost:5000/api/ocr_dec'
# or
# api_url = 'http://0.0.0.0:5000/api/ocr_dec'

response = requests.post(api_url, data=data)
json = response.json()
print(json)
