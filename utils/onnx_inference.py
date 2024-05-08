import numpy as np
import cv2
import onnxruntime as ort


def draw_keypoints(image, keypoints):
    for idx, keypoint in enumerate(keypoints):
        x, y = keypoint
        if x != 0 and y != 0:  # Исключаем точки с нулевыми координатами
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)  # Рисуем круговую метку для ключевой точки
            cv2.putText(image, str(idx + 1), (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1, cv2.LINE_AA) # Рисуем индекс рядом с точкой

    return image


def model_inference(session, input_name, input):
    output = session.run([], {input_name: input})
    return output[0]


def preprocess_img(image, img_size):
    image = cv2.resize(image, (img_size, img_size), cv2.INTER_AREA)
    img = image[:, :, ::-1]
    img = img/255.00
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img,0)
    img = img.transpose(0,3,1,2)
    return img


def single_non_max_suppression(prediction):
    argmax = np.argmax(prediction[4,:])
    x = (prediction.T)[argmax]
    box = x[:4]
    conf = x[4]
    keypts = x[5:]

    return box, conf, keypts


def post_process_single(output):
    box, conf, keypts = single_non_max_suppression(output)
    keypts = smooth_pred(keypts)
    return box, conf, keypts


keypoints_old = None
def smooth_pred(keypoints):
    global keypoints_old
    if keypoints_old is None:
        keypoints_old = keypoints.copy()
        return keypoints

    smoothed_keypoints = []
    for i in range(0, len(keypoints), 3):
        x_keypoint = keypoints[i]
        y_keypoint = keypoints[i+1]
        conf = keypoints[i+2]
        x_keypoint_old = keypoints_old[i]
        y_keypoint_old = keypoints_old[i+1]
        conf_old = keypoints_old[i+2]
        x_smoothed = (conf * x_keypoint + conf_old * x_keypoint_old)/(conf+conf_old)
        y_smoothed = (conf * y_keypoint + conf_old * y_keypoint_old)/(conf+conf_old)
        smoothed_keypoints.extend([x_smoothed, y_smoothed, (conf+conf_old)/2])
    keypoints_old = smoothed_keypoints
    return smoothed_keypoints


def inference_by_img(
        model_path: str,             # путь к весам
        source_path: str,       # путь к изображению
        threshold: float,
        view=True,              # показать ли изображение
        draw_kp=False,          # нарисовать ключевые точки
        is_old_model=False      # при использовании на старой модельке с перепутанными точками
):
    keypoints = []
    session = ort.InferenceSession(model_path, providers=ort.get_available_providers())
    input_name = session.get_inputs()[0].name
    image = cv2.imread(source_path)
    input_img = preprocess_img(image, 640)
    output = model_inference(session, input_name, input_img)
    pps = post_process_single(output[0])
    smoothed = smooth_pred(pps[2])
    for i in range(0, len(smoothed), 3):
        if smoothed[i + 2] > threshold:
            keypoints.append((int(smoothed[i]), int(smoothed[i+1])))



    if is_old_model:
        last_kp = keypoints[-1]
        new_keypoints = list(keypoints[:4]) + [tuple(last_kp)] + list(keypoints[4:])
        keypoints = new_keypoints[:-1]

    if draw_kp:
        image = draw_keypoints(cv2.resize(image, (640, 640), cv2.INTER_AREA), keypoints)

    if view:
        cv2.imshow(f'RTF CHAMPION', image)
        cv2.waitKey(0)

    return keypoints


def resizer(img, keypoints, new_size):
    res_img = cv2.resize(img, new_size, cv2.INTER_AREA)
    res_kpts = np.array(keypoints, dtype=np.float32)
    scale_x = new_size[0] / img.shape[0]
    scale_y = new_size[1] / img.shape[1]
    res_kpts[:, 0] *= scale_x
    res_kpts[:, 1] *= scale_y
    return res_img, res_kpts.astype(int)

if __name__ == '__main__':
    kpts = inference_by_img(
        'dots_0404_ideal.onnx',
        'img/img/000_image_0013079_2020-02-25_13-47-48_roi_001.jpg',
        draw_kp=False,
        is_old_model=False,
        view=False,
        threshold=0.25
    )
    img = cv2.resize(cv2.imread('img/img/000_image_0013079_2020-02-25_13-47-48_roi_001.jpg'), (640, 640), cv2.INTER_AREA)
    new_img, new_kpts = resizer(img, kpts, (128, 128))
    new_img = draw_keypoints(new_img, new_kpts)
    cv2.imwrite('test_resizer.jpg', new_img)

