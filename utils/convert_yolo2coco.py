import os
import json
import time

import cv2
from tqdm import tqdm


def draw_keypoints(image, keypoints):
    for i in range(0, len(keypoints), 3):
        x, y, _ = int(float(keypoints[i])), int(float(keypoints[i+1])), keypoints[i+2]
        if x != 0 and y != 0:  # Исключаем точки с нулевыми координатами
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # Рисуем круговую метку для ключевой точки
            cv2.putText(image, str(int(i/3)+1), (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA) # Рисуем индекс рядом с точкой

    # for idx, keypoint in enumerate(keypoints):
    #     x, y = int(keypoint[0] * image.shape[1]), int(keypoint[1] * image.shape[0])
    #     if x != 0 and y != 0:  # Исключаем точки с нулевыми координатами
    #         cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # Рисуем круговую метку для ключевой точки
    #         cv2.putText(image, str(idx), (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA) # Рисуем индекс рядом с точкой

    return image


def yolo_to_coco(yolo_path, image_dir):
    # Load YOLO annotations and map them to COCO format
    coco_data = {
        'info': {
            'year': '2024',
            'version': '5',
            'description': 'Exported from roboflow.com',
            'contributor': '',
            'url': 'https://public.roboflow.com/object-detection/undefined',
            'date_created': '2024-03-21T08:00:31+00:00'
        },

        'licenses': [{'id': 1, 'url': 'https://creativecommons.org/licenses/by/4.0/', 'name': 'CC BY 4.0'}, ],

        'categories': [
            {
                'id': 0,
                'name': 'cow',
                'supercategory': 'cows',
                'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
                'skeleton': [[1, 2], [2, 3], [1, 4], [4, 6], [6, 9], [9, 11], [3, 5], [5, 8], [8, 10], [10, 11],
                             [12, 2], [12, 7], [7, 11], [1, 12], [3, 12], [7, 9], [7, 10], [4, 12], [6, 12],
                             [12, 5], [8, 12], [6, 7], [8, 7]],
                'images': [],
                'annotations': []
            }
        ],
        'images': [],
        'annotations': []
    }

    images = os.listdir(image_dir)
    images = sorted(images, key=lambda x: int(x.split('.')[0]))

    # Reading YOLO annotation files
    for idx, image_name in enumerate(tqdm(images)):
        im = cv2.imread(image_dir + '/' + image_name)
        height_img,weight_img = im.shape[0], im.shape[1]
        image_id = idx
        image_info = {
            'id': image_id,
            'license': 1,
            'file_name': image_name,
            'height': height_img,
            'width': weight_img,
            'date_captured': '2024-03-21T08:00:31+00:00'
        }
        coco_data['images'].append(image_info)

        with open(os.path.join(yolo_path, str(image_name[:-4]) + ".txt")) as f:
            lines = f.readline().split(" ")
            x, y, w, h = float(lines[1]) * weight_img, float(lines[2]) * height_img, float(lines[3]) * weight_img, float(lines[4]) * height_img
            x_min = int(x - w / 2)
            y_min = int(y - h / 2)
            x_max = int(x + w / 2)
            y_max = int(y + h / 2)
            keypoints = []
            for i, point in enumerate(lines[5:]):
                if point == '2':
                    keypoints.append(int(point))
                elif point == '0.0':
                    keypoints.append(int(float(point)))
                else:
                    if i % 3 == 0:
                        keypoints.append(float(point) * (x_max - x_min))
                    elif i % 3 == 1:
                        keypoints.append(float(point) * (y_max - y_min))
            keypoints = [keypoints[i:i+3] for i in range(0, len(keypoints), 3)]

            # keypoints.insert(3, keypoints[11]) # костыль
            # keypoints = keypoints[:-1]
            keypoints = [item for sublist in keypoints for item in sublist]

            # im = draw_keypoints(im, keypoints)
            # cv2.imshow('test', im)
            # cv2.waitKey(0)
            annotation = {
                'id': len(coco_data['annotations']),
                'image_id': image_id,
                'category_id': int(lines[0]),
                'bbox': [x_min, y_min, x_max, y_max],
                'area': (x_max - x_min) * (y_max - y_min),
                'segmentation': [],
                'iscrowd': 0,
                'keypoints': keypoints
            }

            coco_data['annotations'].append(annotation)


    with open('output.json', 'w') as outfile:
        json.dump(coco_data, outfile)

# Example usage
yolo_to_coco('res/labels', 'res/images')
