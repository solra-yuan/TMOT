import json
import cv2
import numpy as np
from visdom import Visdom

# visdom 서버에 연결 (서버가 실행 중이어야 함)
viz = Visdom(env='coco_visualizer', port=8097, server='http://localhost')
name = 'train_coco_t'

# COCO JSON 파일 경로 (예: coco_file.json)
json_path = f'data/flir_adas_v2/annotations/{name}.json'

# JSON 파일 읽기
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# --- 2. 이미지에 bbox와 segmentation을 그린 후 visdom에 전송 ---
# (COCO JSON의 첫 번째 이미지 사용)
if data.get('images'):
    for i in range(0, 1000, 200):
        image_info = data['images'][i]
        file_name = image_info.get('file_name')

        # 이미지 파일 읽기 (이미지 경로가 올바른지 확인)
        img = cv2.imread(f'data/flir_adas_v2/{name}/{file_name}')
        if img is None:
            print("이미지를 찾을 수 없습니다:", file_name)
        else:
            for annotation in data.get('annotations', []):
                if annotation.get('image_id') == image_info.get('id'):
                    bbox = annotation.get('bbox')

                    x, y, w, h = map(int, bbox)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            viz.image(img_rgb.transpose(2, 0, 1), opts=dict(title=f"Image with BBox and Segmentation {file_name}"))
