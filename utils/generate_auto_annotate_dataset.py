import os
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO


def get_files_in_directory(directory_path, num_files):
    all_files = sorted([os.path.join(directory_path, file) for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))])
    selected_files = [all_files[i] for i in range(start, len(all_files), num_files)]

    return selected_files

def draw_keypoints(image, keypoints):
    for idx, keypoint in enumerate(keypoints.squeeze(0)):
        x, y = int(keypoint[0] * image.shape[1]), int(keypoint[1] * image.shape[0])
        if x != 0 and y != 0:  # Исключаем точки с нулевыми координатами
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # Рисуем круговую метку для ключевой точки
            cv2.putText(image, str(idx), (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA) # Рисуем индекс рядом с точкой

    return image


if __name__ == '__main__':
    img = r"C:\Programming\DataGang\images\identification-train\img"
    step = 54
    start = 2
    output_dir_im = 'res\images'
    output_dir_la = 'res\labels'
    img_files = get_files_in_directory(img, step)
    for i, im in enumerate(tqdm(img_files)):
        orig_img = cv2.imread(im)
        # orig_img = cv2.rotate(orig_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        work_img = cv2.resize(orig_img, (640, 640))
        keypoint_model = YOLO("dots_0404_ideal.pt")
        keypoint_results = keypoint_model(work_img, verbose=False,)
        keypoints = keypoint_results[0].keypoints.xyn
        boxes = keypoint_results[0].boxes.xywhn

        # res_img = draw_keypoints(orig_img, sam_results[0].keypoints.xyn)

        # orig_img = draw_keypoints(orig_img, keypoints)
        # cv2.imshow(f'{id}', orig_img)
        # cv2.waitKey(0)

        cv2.imwrite(f"{Path(output_dir_im)}/{i}.jpg", orig_img)

        with open(f"{Path(output_dir_la)}/{i}.txt", "w") as f:
            f.write("0 ")
            f.write(" ".join(map(lambda x: str(float(x)), boxes[0].reshape(-1))) + " ")
            res_keypoints = []
            for point in keypoints[0]:
                res_keypoints.append(" ".join(map(lambda x: str(float(x)), point)))
            f.write(" 2 ".join(res_keypoints) + " 2")

