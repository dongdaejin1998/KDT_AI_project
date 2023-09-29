import cv2
import numpy as np
import time, os
import sys
import json
import pandas as pd
import multiprocessing

def get_file_list(extension):
    root = os.path.dirname(os.path.abspath(__file__))
    datalist = []
    for path, dirs, files in os.walk(root):
        dir_path = os.path.join(root,path)
        for file in files:
            if os.path.splitext(file)[1] == extension:
                datalist.append(os.path.join(dir_path, file))
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

sentense_label_dict = {}

def preprocessing_video(video, label):
    print("start_preprocessing")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

    
    video_count=0
    os.makedirs('dataset', exist_ok=True)
    os.makedirs('splited', exist_ok=True)

    
    
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
        #cv2.imshow('lip',lip)
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
    fps=cap.get(cv2.CAP_PROP_FPS)

    i = 0
    for info in sentense_info[0]:
        #print(info)
        start_ms = int(info['start_time'] * 1000) # 시작 지점은 상관 없음
        end_ms = int((info['end_time']+0.1) * 1000)  # 0.1초 정도는 버려지는 값 보정을 위해 사용함
        name = os.path.basename(saved_path).split('.')[0]  + f'_{i}.mp4'
        cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)
        saved_path2 = os.path.join('splited',name)
        output = cv2.VideoWriter(saved_path2, fourcc, fps, (96, 96))
        while cap.isOpened():
            ret, frame = cap.read()

            if cap.get(cv2.CAP_PROP_POS_MSEC) >= end_ms:
                break
            
            output.write(frame)

        output.release()
        sentense_label_dict[name] = info['sentence_text']
        i += 1
    cap.release()
    #cv2.destroyAllWindows()
    print(f"[Finish {time.time()-start}]: {file_name} ")


def process_pair(pair):
    video, label = pair
    preprocessing_video(video, label)

if __name__ == '__main__':
    video_list = get_file_list('.mp4')
    label_list = get_file_list('.json')
    pair_list = make_pair_list(video_list, label_list)

    # 멀티프로세싱에 사용할 프로세스 개수
    num_processes = multiprocessing.cpu_count()
    print(num_processes)
    # 멀티프로세싱 작업을 위한 풀 생성
    pool = multiprocessing.Pool(processes=int(num_processes*0.8))

    # pair_list를 작업자 프로세스에 분할하여 작업 실행
    pool.map(process_pair, pair_list)

    # 풀 종료
    pool.close()
    pool.join()

    pool.map(process_pair, pair_list)

    df = pd.DataFrame(list(sentense_label_dict.items()), columns=['Video', 'Sentence'])
    output_label = 'sentence_label.csv'
    df.to_csv(output_label, index=False,encoding='cp949')