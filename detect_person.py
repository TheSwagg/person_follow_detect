import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import os
import numpy as np
from kalman_filter import kalmanFilter
from utils_ import convert_position_to_state, convert_state_to_position, draw_rectangle, calculate_iou, update_trace, draw_trace, draw_text, save_data_to_excel
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TKAgg')
matplotlib.get_backend()

MAX_IOU_THRESHOLD = 0.3
VIDEO_SAVED = True

def detect(save_img=False):
    # 初始化卡尔曼滤波
    kalman_filter = kalmanFilter()
    kalman_filter.initialize()

    trace_list = []
    measurement_list = []
    target_center_list = []

    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # 绘制卡尔曼滤波处理后的边框
                last_x_posterior_estimate = \
                    convert_state_to_position(kalman_filter.last_x_posterior_estimate).astype(np.int16)
                draw_rectangle(im0, last_x_posterior_estimate, color=(255, 0, 0))

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    if names[int(c)] == 'person':   # 只显示person
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    else:   # 只显示person
                        continue    # 只显示person

                # Write results
                max_iou_matched = False
                max_iou = MAX_IOU_THRESHOLD
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        if names[int(cls)] == 'person':     # 只显示person
                            label = f'{names[int(cls)]} {conf:.2f}'
                            # plot_one_box用于在检测nms后将最终的预测bounding box在原图中画出来
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                            # ------------------接下来进行某个人的追踪--------------------

                            # 因为yolov5获取到的边框为tensor的列表类型 所以需要对获取到的边框进行处理
                            position = []
                            for xyxy_ in xyxy:
                                position.append(int(xyxy_.item()))
                                # draw_rectangle(im0, kalman_filter.last_position, color=(255, 0, 255))
                            # print(position)

                            # 绘制所有获取到的person的边框
                            draw_rectangle(im0, position, color=(0, 255, 0))
                            kalman_filter.last_position = \
                                convert_state_to_position(kalman_filter.last_x_posterior_estimate)
                            # 通过上一帧图像的位置和这一帧图像的位置来计算iou
                            iou = calculate_iou(position, kalman_filter.last_position)
                            if iou > max_iou:
                                target_position = position
                                max_iou = iou
                                max_iou_matched = True

                        else:   # 只显示person
                            continue    # 只显示person
                            # 判断target_position是否找到
                if max_iou_matched:
                    # 绘制yolo找到的person的边框（卡尔曼处理前的位置）
                    draw_rectangle(im0, target_position, color=(0, 0, 255))
                    target_measurement = convert_position_to_state(target_position)
                    measurement_list.append(target_measurement[0])
                    # x方向的速度（即为上一帧和这一帧的x方向的位移）
                    velocity_x = target_measurement[0] - kalman_filter.last_x_posterior_estimate[0]
                    # y方向的速度（即为上一帧和这一帧的y方向的位移）
                    velocity_y = target_measurement[1] - kalman_filter.last_x_posterior_estimate[1]
                    target_measurement[4] = velocity_x
                    target_measurement[5] = velocity_y
                    # measurement
                    kalman_filter.measurement = target_measurement
                    # 绘制文字
                    text_position = convert_state_to_position(kalman_filter.measurement)
                    text_center = (int(text_position[0]), int(text_position[1]))
                    # text_center = (50, 50)
                    draw_text(im0, 'tracking', text_center, color=(255, 0, 0))
                    # ----------------预测----------------
                    # x的先验估计值
                    kalman_filter.x_prior_estimate = \
                        np.dot(kalman_filter.a, kalman_filter.last_x_posterior_estimate)
                    # p的先验估计值
                    kalman_filter.p_prior_estimate = \
                        np.dot(np.dot(kalman_filter.a, kalman_filter.last_p_posterior_estimate),
                               kalman_filter.a.T) + \
                        kalman_filter.q
                    # -----------------测量更新-------------------
                    # 卡尔曼滤波增益
                    kalman_filter.k = \
                        np.dot(np.dot(kalman_filter.p_prior_estimate, kalman_filter.h.T),
                               np.matrix(np.dot(np.dot(kalman_filter.h, kalman_filter.p_prior_estimate),
                                                kalman_filter.h.T) + kalman_filter.r).I)
                    # x的后验估计值
                    kalman_filter.x_posterior_estimate = \
                        kalman_filter.x_prior_estimate + np.dot(kalman_filter.k, kalman_filter.measurement -
                                                                np.dot(kalman_filter.h,
                                                                       kalman_filter.x_prior_estimate))
                    # flatten()函数的作用是将矩阵摊平（变为一个一维矩阵）
                    kalman_filter.x_posterior_estimate = np.array(
                        kalman_filter.x_posterior_estimate).flatten()
                    # p的后验估计值
                    kalman_filter.p_posterior_estimate = \
                        np.dot(kalman_filter.i - np.dot(kalman_filter.k, kalman_filter.h),
                               kalman_filter.p_prior_estimate)
                    # 更新跟踪轨迹
                    trace_posterior_estimate = convert_state_to_position(kalman_filter.x_posterior_estimate)
                    target_center = \
                        (int((trace_posterior_estimate[0] + trace_posterior_estimate[2]) / 2),
                         int((trace_posterior_estimate[1] + trace_posterior_estimate[3]) / 2))
                    trace_list = update_trace(target_center, trace_list)
                    target_center_list.append(target_center[0])
                    # ---------------更新数据--------------
                    # 将这一帧的后验估计保存为先验估计 用于下一帧的计算
                    kalman_filter.last_x_posterior_estimate = kalman_filter.x_posterior_estimate
                    kalman_filter.last_p_posterior_estimate = kalman_filter.p_posterior_estimate
                else:
                    # 如果没有找到匹配的person 那么就用上一帧的后验估计值来计算这一帧的后验估计值
                    kalman_filter.x_posterior_estimate = np.dot(kalman_filter.a,
                                                                kalman_filter.last_x_posterior_estimate)
                    # 更新跟踪轨迹
                    trace_posterior_estimate = convert_state_to_position(kalman_filter.x_posterior_estimate)
                    target_center = \
                        (int((trace_posterior_estimate[0] + trace_posterior_estimate[2]) / 2),
                         int((trace_posterior_estimate[1] + trace_posterior_estimate[3]) / 2))
                    trace_list = update_trace(target_center, trace_list)
                    # 绘制文字
                    text_position = convert_state_to_position(kalman_filter.x_posterior_estimate)
                    text_center = (int(text_position[0]), int(text_position[1]))
                    draw_text(im0, 'lost', text_center, color=(0, 0, 255))
                    # ---------------更新数据--------------
                    # 将这一帧的后验估计保存为先验估计 用于下一帧的计算
                    kalman_filter.last_x_posterior_estimate = kalman_filter.x_posterior_estimate
                draw_trace(im0, trace_list)
                draw_text(im0, 'all objects(green)', (30, 40), color=(0, 255, 0))
                draw_text(im0, 'target(red)', (30, 70), color=(0, 0, 255))
                draw_text(im0, 'last posterior estimate(blue)', (30, 100), color=(255, 0, 0))

                print(target_center_list)
                print(measurement_list)
                x = []
                for i in range(len(target_center_list)):
                    x.append(i)
                y = target_center_list
                y2 = measurement_list
                plt.plot(x, y)
                plt.plot(x, y2)
                # plt.show()

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
        plt.savefig('pic_2.png')

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 权重：训练好的网络模型
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    # 检测图片的文件夹路径
    parser.add_argument('--source', type=str, default='data/videos/person-follow.mp4', help='source')  # file/folder, 0 for webcam
    # 测试图像resize
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # confidence 置信度
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    # NMS 非最大值抑制
    parser.add_argument('--iou-thres', type=float, default=0.2, help='IOU threshold for NMS')
    # cuda
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 是否显示结果
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # 保存的类别
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    # 增强的nms
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # 去除不必要的部分的参数 只保留预测部分所需要的参数
    parser.add_argument('--update', action='store_true', help='update all models')
    # 预测结果所保存的位置
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    # 预测结果所保存的名称
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
