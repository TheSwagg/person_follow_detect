import os
import cv2
import xlwt
import numpy as np

def convert_position_to_state(position):            # position[left_up_x, left_up_y, right_down_x, right_down_y]
    center_x = (position[0] + position[2]) / 2      # 中心点x坐标
    center_y = (position[1] + position[3]) / 2      # 中心点y坐标
    width = position[2] - position[0]               # width
    height = position[3] - position[1]              # height
    return np.array([center_x, center_y, width, height, 0, 0])

def convert_state_to_position(state):               # state[center_x, center_y, width, height, 0, 0]
    left_up_x = state[0] - state[2] / 2             # 左上角的x坐标（中心点x坐标 - 宽的一半）
    left_up_y = state[1] - state[3] / 2             # 左上角的y坐标（中心点y坐标 - 高的一半）
    right_down_x = state[0] + state[2] / 2          # 右下角的x坐标（中心点x坐标 + 宽的一半）
    right_down_y = state[1] + state[3] / 2          # 右下角的y坐标（中心点y坐标 + 高的一半）
    return np.array([left_up_x, left_up_y, right_down_x, right_down_y])

def draw_rectangle(image, points, color=(0, 0, 0)): # 绘制矩形
    cv2.rectangle(image, (points[0], points[1]), (points[2], points[3]), color)

def calculate_iou(rectangle_1, rectangle_2):        # 计算iou
    left_up_x_1, left_up_y_1, right_down_x_1, right_down_y_1 = rectangle_1
    left_up_x_2, left_up_y_2, right_down_x_2, right_down_y_2 = rectangle_2
    # 计算矩形的面积
    rectangle_area_1 = (right_down_x_1 - left_up_x_1) * (right_down_y_1 - left_up_y_1)
    rectangle_area_2 = (right_down_x_2 - left_up_x_2) * (right_down_y_2 - left_up_y_2)
    # 计算两个矩形集合部分的宽度，高度以及面积
    intersection_width = max(min(right_down_x_1, right_down_x_2) - max(left_up_x_1, left_up_x_2), 0)
    intersection_height = max(min(right_down_y_1, right_down_y_2)- max(left_up_y_1, left_up_y_2), 0)
    intersection_area = intersection_width * intersection_height
    # 计算两个矩形不重合部分的面积
    uniou_area =  rectangle_area_1 + rectangle_area_2 - intersection_area
    # 计算iou（iou = 交集和并集的比值）
    iou = intersection_area / uniou_area
    return iou

def update_trace(rectangle_center, trace_list, max_length=50):          # 更新轨迹（最大长度为50）
    if len(trace_list) <= max_length:
        trace_list.append(rectangle_center)
    else:
        trace_list.pop(0)
        trace_list.append(rectangle_center)
    return trace_list

def draw_trace(image, trace_list, color=(255, 255, 255), thickness=3):      # 绘制轨迹
    for index, item in enumerate(trace_list):
        if index < 1:
            continue
        cv2.line(image, (trace_list[index][0], trace_list[index][1]),
                 (trace_list[index - 1][0], trace_list[index - 1][1]), color, thickness)

def draw_text(image, text, rectangle_center, font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7,
              color=(0, 0, 0), thinkness=2):        # 在指定位置书写文字
    cv2.putText(image, text, rectangle_center, font_face, font_scale, color, thinkness)

def save_data_to_excel(data_matrix, filepath):
    if os.path.exists(filepath):
        os.remove(filepath)
    workbook = xlwt.Workbook()
    worksheet1 = workbook.add_sheet('sheet1', cell_overwrite_ok=True)
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = '宋体'
    font.color = 'black'
    font.height = 220
    style.font = font
    alignment = xlwt.Alignment()
    alignment.vert = xlwt.Alignment.VERT_CENTER
    style.alignment = alignment
    for idx0 in range(data_matrix.shape[0]):
        for idx1 in range(data_matrix.shape[1]):
            worksheet1.write(idx0, idx1, label=float(data_matrix[idx0, idx1]), style=style)
    workbook.save(filepath)