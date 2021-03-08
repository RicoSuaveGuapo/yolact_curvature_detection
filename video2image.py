import cv2 
import os
import time
import multiprocessing as mp

def read_video(video_path):
    video = cv2.VideoCapture(video_path)
    video_name = video_path.split('/')[-1][:-4]
    print(f'{video_name} are cpatured')
    
    if not os.path.exists(f'data/{video_name}'):
        os.mkdir(f'data/{video_name}')

    # frame 
    currentframe = 0
    while(True):     
        # reading from frame 
        ret, frame = video.read()
        # ret代表成功與否（True 代表成功，False 代表失敗）, frame 就是攝影機的單張畫面
        if ret: 
            # if video is still left continue creating images 
            frame_name = currentframe
            if frame_name < 10:
                name = f'./data/{video_name}/frame' + '000' + str(currentframe) + '.jpg'
                cv2.imwrite(name, frame) 
                currentframe += 1
            elif frame_name < 100:
                name = f'./data/{video_name}/frame' + '00' + str(currentframe) + '.jpg'
                cv2.imwrite(name, frame) 
                currentframe += 1
            elif frame_name < 1000:
                name = f'./data/{video_name}/frame' + '0' + str(currentframe) + '.jpg'
                cv2.imwrite(name, frame) 
                currentframe += 1
            else:
                name = f'./data/{video_name}/frame' + str(currentframe) + '.jpg'
                cv2.imwrite(name, frame) 
                currentframe += 1
        else: 
            break
  
    # Release all space and windows once done 
    video.release() 
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    start_time = time.time()
    video_par_path = '/home/rico-li/Job/豐興鋼鐵/high_quality_video'
    video_dirs = os.listdir(video_par_path)
    video_dirs = [os.path.join(video_par_path, video_dir) for video_dir in video_dirs if video_dir.split('/')[-1] != '.DS_Store']
    video_paths = [os.path.join(video_dir, video_path) for video_dir in video_dirs for video_path in os.listdir(video_dir)]
    video_count = len(video_paths)
    print(f'there are {video_count} videos')
    print(f'has {os.cpu_count()} cpus')
    
    try: 
        if not os.path.exists('data'): 
            os.makedirs('data')
    except OSError: 
        print ('Error: Creating directory of data') 

    video_idx = 0
    counts = video_count//os.cpu_count()
    left = video_count%os.cpu_count()
    now = video_count
    for _ in range(counts+1):
        processes = []
        if now >= os.cpu_count():
            for _ in range(os.cpu_count()):
                p = mp.Process(target=read_video, args=[video_paths[video_idx]])
                p.start()
                processes.append(p)
                video_idx += 1
            for process in processes:
                process.join()
        else:
            for _ in range(left):
                p = mp.Process(target=read_video, args=[video_paths[video_idx]])
                p.start()
                processes.append(p)
                video_idx += 1
            for process in processes:
                process.join()
        now -= os.cpu_count()
    print(f'\n--- spend {time.time() - start_time:.2f} sec ---\n')

  
