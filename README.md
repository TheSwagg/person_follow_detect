# person_follow_detect
基于卡尔曼滤波的行人检测以及跟踪(person_follow and person_detect)  
运行：直接右键运行 detect_person.py 即可  
卡尔曼滤波初始化及参数在 kalman_filter.py 中  
模型：就是使用了 yolov5s.pt，只不过把除了 person 都处理掉了而已  
卡尔曼滤波处理后的图像会保存在 pic_2.png 中
