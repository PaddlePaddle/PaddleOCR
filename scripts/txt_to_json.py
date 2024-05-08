import json


TXT_PATH=r"E:\projects\ballooning\dataset\layout\label.txt"
SAVE_PATH = r"E:\projects\ballooning\dataset\layout\train.json"


MAP_CLS = {
    "content": 0,
    "table": 1,
    "text":2
}

dataset = {
    "images": [],
    "annotations": [],
    "categories": [
        {"supercategory": "", "id": 0, "name": "content"}, 
        {"supercategory": "", "id": 1, "name": "table"}, 
        {"supercategory": "", "id": 2, "name": "text"}
    ]
}



# 打开文件
with open(TXT_PATH, 'r', encoding='utf-8') as file:
    # 按行读取内容
    lines = file.readlines()

# 输出每一行内容
bbox_cnt = 0
for idx, line in enumerate(lines):
    image_path =   line.split("\t")[0]
    annotations =   line.split("\t")[1]
    
    dataset["images"].append({
            "file_name": image_path.split("/")[1],
            "height": 2339,
            "width": 3308,  
            "id": idx
    })

    bbox_list = json.loads(annotations)
    for bbox_item in bbox_list:
        x1 = bbox_item["points"][0][0]
        y1 = bbox_item["points"][0][1]
        x2 = bbox_item["points"][2][0]
        y2 = bbox_item["points"][2][1]

        dataset["annotations"].append({
            "iscrowd": 0, 
            "image_id": idx,  
            "bbox": [x1,y1,x2-x1+1,y2-y1+1],
            "area": (x2-x1+1)*(y2-y1+1),
            "category_id": MAP_CLS[bbox_item["key_cls"]],   
            "id": bbox_cnt      
        })
        bbox_cnt += 1


# Write the dictionary into the JSON file
with open(SAVE_PATH, "w") as json_file:
    json.dump(dataset, json_file, indent=4)  # The indent parameter is optional for pretty formatting