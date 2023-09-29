import cv2
import numpy as np
import time, os
import sys
import json
import pandas as pd
import ray


sentense_label_dict = {}

def get_file_list(addres, extension):
    root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root, addres)
    datalist = []

    for path, dirs, files in os.walk(data_path):
        for file in files:
            if os.path.splitext(file)[1] == extension:
                datalist.append(os.path.join(path, file))
    
    return datalist

def make_pair_list(list1, list2):
    pair_list = []
    for file_path1 in list1:
        for file_path2 in list2:
            if os.path.basename(file_path1).split('.')[0] == os.path.basename(file_path2).split('.')[0]:
                pair_list.append([file_path1,file_path2])
            else:
                pass
    return pair_list



def get_lip_data(file_path):
    lip_data = []
    sentence_info = []
    with open(file_path, "r", encoding="UTF8") as f:
        contents = f.read()
        json_data=json.loads(contents)
        lip_data.append(json_data[0]["Bounding_box_info"]["Lip_bounding_box"]["xtl_ytl_xbr_ybr"])
        sentence_info.append(json_data[0]["Sentence_info"])
    return lip_data, sentence_info

@ray.remote
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



@ray.remote
def preprocessing_video(video, label):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

    
    video_count=0


    
    
    start = time.time()
    print(f"video: {os.path.basename(video)}")
    print(f'label: {os.path.basename(label)}')
    
    # step 1: 입술 부분만 가져온 뒤 흑백 처리 후 저장
    cap = cv2.VideoCapture(video)
    
    width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps=cap.get(cv2.CAP_PROP_FPS)
    
    file_name = os.path.basename(video).split('.')[0] + '_preprocessed.mp4'
    saved_path = os.path.join('dataset',file_name)
    out = cv2.VideoWriter(saved_path, fourcc, fps, (96, 96),isColor = False)

    frame_count=0
    lib_data, sentense_info = get_lip_data(label)

    

    while cap.isOpened():
        ret,frame=cap.read()
        if not ret:
            break
        lip=frame.copy()
        lip=lip[lib_data[0][frame_count][0]:lib_data[0][frame_count][2],lib_data[0][frame_count][1]:lib_data[0][frame_count][3]]
        frame_count+=1

        lip = cv2.resize(lip, dsize=(96,96))
        lip = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
        


        out.write(lip)

        #cv2.imshow('frame',frame)
        #cv2.imshow('norm_lip',lip)
        if cv2.waitKey(1)& 0xFF == ord('q'):
            break
    video_count+=1

    cap.release()
    out.release()
    #cv2.destroyAllWindows()
    

    # step2: 문장 별로 자르기

    cap = cv2.VideoCapture(saved_path)
    width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps=30
    target_fps=540

    i = 0
    for info in sentense_info[0]:
        #print(info)
        start_ms = int(info['start_time'] * 1000) # 시작 지점은 상관 없음
        end_ms = int((info['end_time']+0.1) * 1000)  # 0.1초 정도는 버려지는 값 보정을 위해 사용함
        name = os.path.basename(saved_path).split('.')[0]  + f'_{i}.mp4'
        cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)
        saved_path2 = os.path.join('splited',name)
        output = cv2.VideoWriter(saved_path2, fourcc, fps, (96, 96),isColor = False)
        while cap.isOpened():
            ret, frame = cap.read()

            if cap.get(cv2.CAP_PROP_POS_MSEC) >= end_ms:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            output.write(frame)
        
        # padding으로 채워진 영상을 생성하여 저장
        padding_frames = target_fps - count  # 목표 FPS(540) 기준으로 추가로 필요한 프레임 수 계산
        for _ in range(padding_frames):
            blank_frame = np.zeros((96, 96), dtype=np.uint8)  # 검정색(0)으로 채워진 프레임 생성
            output.write(blank_frame)
        output.release()
        sentense_label_dict[name] = info['sentence_text']
        i += 1
    cap.release()
    #cv2.destroyAllWindows()

    
    print(f"[Finish {time.time()-start}]: {file_name} ")



@ray.remote
def mean_std_processing(video,mean,std):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

    cap = cv2.VideoCapture(video)
    fps=cap.get(cv2.CAP_PROP_FPS)
    width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    output = cv2.VideoWriter(video, fourcc, fps, (96, 96),isColor = False)
    while cap.isOpened():

        ret,frame=cap.read()

        if not ret:
            break
        frame=frame.astype(np.float32)
        frame=frame/255.0
        image_normalized=(frame-mean/std) 
        output.write(image_normalized)

    output.release()
    cap.release()



if __name__ == '__main__':

    ray.init(num_cpus=12)

    video_list = get_file_list('data','.mp4')
    label_list = get_file_list('data','.json')


    
    pair_list = make_pair_list(video_list, label_list)

    os.makedirs('dataset', exist_ok=True)
    os.makedirs('splited', exist_ok=True)
    # 멀티프로세싱 작업을 위한 풀 생성
    ray_results1 = [preprocessing_video.remote(video, label) for video, label in pair_list]
    ray.get(ray_results1)


    dataset_list = get_file_list('splited','.mp4')
    mean_std_results = mean_std.remote(dataset_list)
    mean, std = ray.get(mean_std_results)
    mean=round(mean/255,3)
    std=round(std/255,3)
    print(mean,std)
    f = open("mean_std.txt",'w')
    f.write(str(mean))
    f.write("=====\n")
    f.write(str(std))
    f.close()

    splitd_list=get_file_list('splited','.mp4')
    ray_results2 = [mean_std_processing.remote(video, mean,std) for video in splitd_list]
    ray.get(ray_results2)

    ray.shutdown()



    df = pd.DataFrame(list(sentense_label_dict.items()), columns=['Video', 'Sentence'])
    output_label = 'sentence_label.csv'
    df.to_csv(output_label, index=False,encoding='cp949')