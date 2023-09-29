import cv2
import numpy as np
import time, os
import sys
import json
import pandas as pd
import ray

def get_file_list(addres, extension):
    root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root, addres)
    datalist = []

    for path, dirs, files in os.walk(data_path):
        for file in files:
            if os.path.splitext(file)[1] == extension:
                datalist.append(os.path.join(path, file))
    
    return datalist


@ray.remote
def count_fps_video_parallel(video_list):
    count_fps = 0
    fps_list = []
    print("counting")

    for video in video_list:
        start = time.time()
        #print(f"video: {os.path.basename(video)}")
        
        # step 1: 입술 부분만 가져온 뒤 흑백 처리 후 저장
        cap = cv2.VideoCapture(video)
        
        fps = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps_list.append(fps)
        count_fps += fps
        
    print(max(fps_list))
    print(count_fps)
    print(len(video_list))
    print(count_fps // len(video_list))
    
    return max(fps_list), count_fps, len(video_list), count_fps // len(video_list)






def mean_std(videos):
    frame_count = 0
    mean_values = []
    std_values = []
    mean_average=0
    std_average=0

    for video in videos:
        cap = cv2.VideoCapture(video)

        while cap.isOpened():
            ret,frame=cap.read()
            
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            mean,std=cv2.meanStdDev(frame)

            mean_value=mean.flatten()[0]
            std_value=std.flatten()[0]

            mean_values.append(mean_value)
            std_values.append(std_value)

        mean_average += np.mean(mean_values)
        std_average += np.mean(std_values)
        
    mean_final=mean_average/len(videos)
    std_final=std_average/len(videos)

    return mean_final,std_final


if __name__ == '__main__':
    ray.init()

    splitd_list = get_file_list('splited', '.mp4')

    # count_fps_video 함수를 병렬 처리하기 위해 remote 함수로 호출
    mean_std_results = count_fps_video_parallel.remote(splitd_list)
    max_fps, total_fps, num_videos, avg_fps = ray.get(mean_std_results)

    print(f"Max FPS: {max_fps}")
    print(f"Total FPS: {total_fps}")
    print(f"Number of Videos: {num_videos}")
    print(f"Average FPS: {avg_fps}")
    
    ray.shutdown()


    #mean, std = mean_std(splitd_list)
    #mean = round(mean, 3)
    #std = round(std, 3)
    #print(mean, std)

    
    setence_label = pd.read_csv("setence_label.csv", encoding='cp949')
    setence = setence_label['Sentence']
