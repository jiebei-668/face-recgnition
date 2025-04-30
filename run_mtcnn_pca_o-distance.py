import os
from datetime import datetime

import cv2
import argparse
from mtcnn import MTCNN
from PIL import Image
import numpy as np
from pathlib import Path
from mtcnn_pytorch.src import visualization_utils
os.environ['KMP_DUPLICATE_LIB_OK']='True'


if __name__ == "__main__":
    # =================================================参数设定===============================================================

    # 解析器
    parser = argparse.ArgumentParser(description='use mtcnn to detect face, use pca and o-dis to recognize face')
    parser.add_argument('--video_name', default='IPartment1.mp4', type=str, help='the dataset source')
    parser.add_argument('--video_dir',  default='video_for_test', type=str, help='测试视频所在目录')
    parser.add_argument('--out_video_path', default='out_with_detection_and_recognition/test1.mp4', type=str, help='')
    args = parser.parse_args()

    # ======================================================================================================================
    # ============================================视频路径与图片保存路径=========================================================

    # 绝对路径
    ROOT_PATH = os.path.abspath(os.getcwd())  # this is a file path

    # 视频路径
    video_data_path = Path(os.path.join(ROOT_PATH, args.video_dir, args.video_name))
    if not video_data_path.exists():
        raise FileNotFoundError("{}没有放在{}目录下".format(args.video_name, args.video_dir))

    # 创建输出视频的目录
    out_video_path_abs = os.path.abspath(args.out_video_path)
    if not Path(out_video_path_abs).exists():
        os.makedirs(out_video_path_abs, exist_ok=True)


    # ======================================================================================================================
    # ============================================视频对象与其信息=============================================================
    # 获取video
    video = cv2.VideoCapture(str(video_data_path))

    # 视频帧率
    fps = video.get(cv2.CAP_PROP_FPS)

    # 视频总帧数
    frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)

    # 视频宽度,视频高度
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # 初始化视频写入器（如果需要保存）
    video_out = None
    out = None
    if args.out_video_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(args.out_video_path, args.video_name), fourcc, fps, size)

    # 打印视频信息
    print("视频信息如下:")
    print("帧率:{0}\n帧数:{1}\n帧像素:{2}x{3}\n".format(fps,frameCount,size[0],size[1]))

    # 读取第一帧
    success, frame = video.read()

    # ======================================================================================================================
    # ============================================模型声明====================================================================

    # 模型声明
    mtcnn = MTCNN()  # ValueError: Object arrays cannot be loaded when allow_pickle=False ->pip install numpy==1.16.2

    # ======================================================================================================================
    # ============================================视频帧人脸检测===============================================================

    i = 0
    # 对每帧进行人脸检测
    while success:
        try:
            p = Image.fromarray(frame[..., ::-1])  # rgb ->bgr
            boxes, faces,landmarks = mtcnn.align_multi(p)  # bgr -> rgb
            #  把每张脸保存
            # for face in faces:
            #     now_date_time = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
            #     filename = str(save_path) + "/" + now_date_time + ".jpg"
            #     print("{} frame:face captured保存第{}帧".format(i,i))
            #     face.save(filename)

            # handle_frame = visualization_utils.draw_boxes_and_landmarks_on_image(p,boxes,landmarks)  # 对视频帧人脸画框
            handle_frame = visualization_utils.draw_boxes_and_class_on_image(p,boxes,class_name="qiangping")  # 对视频帧人脸画框
            draw_frame = np.array(handle_frame)[...,::-1]  # rgb->bgr
            # cv2.imshow(" ",draw_frame)
            # cv2.waitKey(0)
            # 写入输出视频
            if out is not None:
                out.write(draw_frame)
            # 画框视频帧保存
        except Exception as e:
            # 没有检测到人脸视频帧保存
            filename = str(save_path / '{}.jpg'.format(i))
            # print('{} frame:no face captured'"保存第{}帧".format(i,i))
            print("error[" + str(e) + "]")
            # cv2.imencode('.jpg', frame)[1].tofile(filename)
        success, frame = video.read()

    print("finish!")
    # 释放视频资源
    video.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
