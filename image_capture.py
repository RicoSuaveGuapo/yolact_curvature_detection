import matplotlib.pyplot as plt
import multiprocessing as mp
import matplotlib
import numpy as np
import shutil
import time
import cv2
import os

from preprocess import image_crop, image_back

def capture(video_path:str, video_bol:bool, avg_step=2, threshold=0.3, type_name='U150', extend_frame:int=None):
    # (h, w, c)
    if video_bol == True:
        video = cv2.VideoCapture(video_path)
        frame_index = 0
        while(video.isOpened()):
            ret, frame = video.read()
            frame = image_crop(frame)
            if (ret) and (np.abs(np.median(frame[-1,...])) < threshold):
                frame_index += 1
            elif (ret) and (np.abs(np.median(frame[-1,...])) > threshold):
                break
            else:
                raise IOError('reach end of video, do not activate at all')
        return frame_index
    else:
        image_paths = os.listdir(video_path)
        image_paths.sort()
        image_paths = [os.path.join(video_path, image_path) for image_path in image_paths]
        median_list = []
        capture_idx = []
        median_diff = []
        for frame_index, image_path in enumerate(image_paths):
            frame = image_crop(image_path)
            frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, np.ones((2,2))) # denoising
            # activate region:
            # (80,350), (500, 380)
            frame = np.median(frame[339:379, 80:500, :])
            median_list.append(frame)
            if frame_index < avg_step:
                continue
            else:
                median_diff.append(np.mean(median_list[-avg_step-1:-1]) - np.mean(median_list[-avg_step:]))
                if abs(median_diff[-1]) < threshold:
                    pass
                else:
                    capture_idx.append(frame_index)
        assert len(capture_idx) != 0, 'reach end of video, do not activate at all'

        capture_median = [median_diff[idx-avg_step] for idx in capture_idx] # since the difference of median_diff's idx and meidan_list's idx is avg_step
        if capture_median[0] < 0:
            print('\n--- Thick one ---\n')
            capture_idx = [idx for i, idx in enumerate(capture_idx) if capture_median[i] > 0.5]

            # drop duplicate within 1 frames (1/25 sec), and keep the last ones
            capture_idx = capture_idx[::-1]
            capture_idx = [idx for i, idx in enumerate(capture_idx) if capture_idx[i-1] - capture_idx[i] != 1]
            capture_idx = capture_idx[::-1]
        else:
            print('\n--- Thin one ---\n')
            # print([capture_median[i] for i, idx in enumerate(capture_idx) if capture_median[i] < -(threshold)])
            capture_idx = [idx for i, idx in enumerate(capture_idx) if capture_median[i] < -(threshold)]
        
            # drop duplicate within 25 frames (1 sec), and keep the last ones
            capture_idx = capture_idx[::-1]
            capture_idx = [idx for i, idx in enumerate(capture_idx) if capture_idx[i-1] - capture_idx[i] > 25]
            capture_idx = capture_idx[::-1]

        video_name = video_path.split('/')[-1]
        if extend_frame == None:
            des = os.path.join('data/clean_data', type_name, video_name)
            if os.path.exists(des) == False:
                os.mkdir(des)
            idx_paths = [image_paths[idx] for idx in capture_idx]
            move_action = [shutil.copy(idx_path, os.path.join(des, idx_path.split('/')[-1])) for idx_path in idx_paths]
        else:
            des = os.path.join(f'data/clean_data_{extend_frame}frames', type_name, video_name)
            if os.path.exists(des) == False:
                os.mkdir(des)

            # expand the capture_idx in order to increase the data set
            exp_capture_idx = []
            for idx in capture_idx:
                exp_capture_idx += [idx-i for i in range(0,extend_frame+1)]
                exp_capture_idx += [idx+i for i in range(1,extend_frame+1)]
                exp_capture_idx.sort()

            idx_paths = [image_paths[idx] for idx in exp_capture_idx]
            move_action = [shutil.copy(idx_path, os.path.join(des, idx_path.split('/')[-1])) for idx_path in idx_paths]


        # fig, ax = plt.subplots(2,1, sharex=True)
        # fig.set_figheight(10)
        # fig.set_figwidth(20)
        # title_name = video_name
        # fig.suptitle(title_name)
        # ax[0].plot(median_list[10:])
        # ax[0].set_title('Pixel Value')
        # ax[1].plot(median_diff)
        # ax[1].set_title('Pixel Difference Value')
        # plt.savefig(f'EDA/Pixel_median/{video_name}.png')

        # print(idx_paths)
        # print(capture_idx)
        # print(f'find {len(capture_idx)} images')
        # return capture_idx, median_list, median_diff


if __name__ == "__main__":
    path = 'data/A30'
    image_dirs = os.listdir(path)
    image_dirs = [os.path.join(path, image_dir) for image_dir in image_dirs]
    start_time = time.time()
    avg_step = 10
    threshold = 0.25
    extend_frame = 20

    video_count = len(image_dirs)
    print(f'there are {video_count} videos')
    print(f'has {os.cpu_count()} cpus')
    
    try: 
        if os.path.exists(os.path.join(f'data/clean_data_{extend_frame}frames', path.split('/')[-1])) == False: 
            os.makedirs(os.path.join(f'data/clean_data_{extend_frame}frames', path.split('/')[-1]))
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
                p = mp.Process(target=capture, args=[image_dirs[video_idx], False, avg_step, threshold, path.split('/')[-1], extend_frame])
                p.start()
                processes.append(p)
                video_idx += 1
            for process in processes:
                process.join()
        else:
            for _ in range(left):
                p = mp.Process(target=capture, args=[image_dirs[video_idx], False, avg_step, threshold, path.split('/')[-1], extend_frame])
                p.start()
                processes.append(p)
                video_idx += 1
            for process in processes:
                process.join()
        now -= os.cpu_count()

    print(f'\n--- spend {time.time() - start_time:.2f} sec ---\n')
    # act_index, median_list, median_diff = capture(image_dir, False, avg_step=avg_step,threshold=threshold)
    

    # fig, ax = plt.subplots(2,1, sharex=True)
    # fig.set_figheight(10)
    # fig.set_figwidth(20)
    # title_name = image_dir.split('/')[-1]
    # fig.suptitle(title_name)
    # ax[0].plot(median_list[10:])
    # ax[0].set_title('Pixel Value')
    # ax[1].plot(median_diff)
    # ax[1].set_title('Pixel Difference Value')
    # plt.savefig('3.8t_1528_3_pixel_median_2.png')
    # plt.show()

    # --- using non-rectangle box -- 
    # mask = np.zeros(image.shape, dtype=np.uint8)
    # roi_corners = np.array([[(10,10), (300,300), (10,300)]], dtype=np.int32)
    # # fill the ROI so it doesn't get wiped out when the mask is applied
    # channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    # ignore_mask_color = (255,)*channel_count
    # cv2.fillPoly(mask, roi_corners, ignore_mask_color)

    # decide the range as activate zone
    # --- using (80,350), (500, 380) --- 
    # img_target = 'data/20200817_A30x4t_1613_直度ok_13支/frame3163.jpg'
    # img_target = image_crop(img_target)
    # img_target = cv2.rectangle(img_target, (80,350), (500, 380), (0,255,0), 2)
    # # img_target = img_target[339:379, 80:500, :]
    # fig, ax = plt.subplots(1,1)
    # fig.set_figheight(10)
    # fig.set_figwidth(10)
    # ax.imshow(img_target.astype(np.uint8))
    # plt.show()

