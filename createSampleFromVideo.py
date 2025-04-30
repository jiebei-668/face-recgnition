import os
from ast import parse

import cv2
from mtcnn import MTCNN
from PIL import Image
import numpy as np
from mtcnn_pytorch.Utils import draw_boxes_and_landmarks_on_image, \
    draw_boxes_and_class_on_image

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
from datetime import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, help="path to video file")
    parser.add_argument("--img_output_dir", type=str, help="path to output directory")
    parser.add_argument("--video_output_dir", type=str, help="path to output directory")
    parser.add_argument("--video_output_name", type=str, help="path to output directory")


    args = parser.parse_args()
    img_output_dir = args.img_output_dir
    video_path = args.video_path
    video_output_dir = args.video_output_dir
    video_output_name = args.video_output_name
    if img_output_dir is not None:
        os.makedirs(img_output_dir, exist_ok=True)
    if video_output_dir is not None:
        os.makedirs(video_output_dir, exist_ok=True)
    # 初始化MTCNN模型
    mtcnn = MTCNN()

    # 打开视频文件

    cap = cv2.VideoCapture(video_path)

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 初始化视频写入器（如果需要保存）
    video_out = None
    out = None
    if video_output_dir is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(video_output_dir, video_output_name), fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        # cv2.imshow('frame', frame)
        # cv2.waitKey(0)
        if not ret:
            break

        # 转换颜色空间 BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            # 人脸检测
            boxes, landmarks = mtcnn.detect_faces(Image.fromarray(rgb_frame), min_face_size=300)

            if boxes is not None and len(boxes) != 0:
                # 绘制检测结果
                # draw_img = draw_boxes_and_landmarks_on_image(
                #     Image.fromarray(rgb_frame),
                #     boxes,
                #     landmarks
                # )
                draw_img = draw_boxes_and_class_on_image(
                    Image.fromarray(rgb_frame),
                    boxes,
                    landmarks,
                    class_name="qiangping"
                )
                # cv2.imshow("testtest", draw_img)
                # cv2.waitKey(0)
                # 只把人脸所在的框搞出来
                face = None
                img_name = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3] + ".jpeg"
                for box in boxes:
                    start_x, start_y, end_x, end_y, _ = boxes[0]
                    face = frame[round(start_y):round(end_y), round(start_x):round(end_x), :].copy()
                    # cv2.imshow('face', face)
                    cv2.imwrite(os.path.join(img_output_dir, img_name), face)
                    # cv2.waitKey(0)
                # 转换回BGR格式用于显示
                bgr_img = cv2.cvtColor(np.array(draw_img), cv2.COLOR_RGB2BGR)
                # 显示实时结果
                # cv2.imshow('Video Face Detection', bgr_img)

                # 写入输出视频
                if out is not None:
                    out.write(bgr_img)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            continue

    # 释放资源
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
