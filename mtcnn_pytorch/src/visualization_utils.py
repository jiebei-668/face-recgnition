import numpy as np
import cv2

def draw_boxes_and_landmarks_on_image(image, boxes, landmarks):
    if len(boxes)!=0 and len(landmarks)!=0:
        image = np.array(image)
        # plt.imshow(image)
        # plt.show()
        for box, landmark in zip(boxes, landmarks):
            x1, y1, x2, y2, score = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            score = round(score, 2)

            # 在图像上绘制方框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # 显示概率大小
            text = f'Probability: {score}'
            text_position = (x1, y1 - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (0, 255, 0)
            thickness = 1
            cv2.putText(image, text, text_position, font, font_scale, color, thickness)

            # 绘制特征点
            landmark_reshaped = landmark.reshape(5, 2)
            for point in landmark_reshaped:
                x, y = int(point[0]), int(point[1])
                cv2.circle(image, (x, y), 1, (0, 0, 255), 1)


        return image

def draw_boxes_and_class_on_image(image, boxes, class_name="qiangping"):
    image = np.array(image)
    if len(boxes)!=0:
        # plt.imshow(image)
        # plt.show()
        for box in boxes:
            x1, y1, x2, y2, score = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            score = round(score, 2)

            # 在图像上绘制方框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # 显示概率大小
            # text = f'Probability: {score}'
            # text_position = (x1, y1 - 10)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # font_scale = 0.5
            # color = (0, 255, 0)
            # thickness = 1
            # cv2.putText(image, text, text_position, font, font_scale, color, thickness)

            # 绘制特征点
            # landmark_reshaped = landmark.reshape(5, 2)
            # for point in landmark_reshaped:
            #     x, y = int(point[0]), int(point[1])
            #     cv2.circle(image, (x, y), 1, (0, 0, 255), 1)

            # 绘制类别名称
            # 绘制类别名称（在方框左上角）
            class_text = class_name

            # 设置字体参数
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            color = (0, 0, 255)  # 红色文字
            thickness = 2

            # 计算文字尺寸
            (text_width, text_height), _ = cv2.getTextSize(class_text, font, font_scale, thickness)

            # 绘制文字背景（可选）
            bg_color = (0, 255, 0)  # 绿色背景
            cv2.rectangle(image,
                          (x1, y1 - text_height - 5),  # 背景位置
                          (x1 + text_width + 5, y1),
                          bg_color,
                          -1)

            # 绘制文字
            cv2.putText(image,
                        class_text,
                        (x1, y1 - 5),
                        font,
                        font_scale,
                        color,
                        thickness)

        return image
