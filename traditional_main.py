import os
import cv2
import math
import time
import numpy as np
import matplotlib.pyplot as plt

from preprocess import image_crop, residual

def hough(image_edage, r, theta, threshold, original_image, r_supper, plot=False):
    lines = cv2.HoughLines(img_edage,r,theta,threshold)
    img = original_image.copy()
    if lines is not None:
        lines = lines.squeeze()
        # apply rough criteria
        # theta < 30 degree
        # r > 200
        angle_all = [line[-1] for line in lines if (line[-1] < 30 * np.pi/180) and (line[0] > 200)]
        r_all = [line[0] for line in lines if (line[-1] < 30 * np.pi/180) and (line[0] > 200)]

        # nearby suppresion
        # sorted by r
        angle_all = [remain for _, remain in sorted(zip(r_all, angle_all))]
        r_all.sort()
        # keep the left/right-most
        maxmin_r = [r_all[0]] + [r_all[-1]]
        maxmin_angle = [angle_all[0]] + [angle_all[-1]]
        r = r_all[1:-1]
        angle = angle_all[1:-1]
        if len(r) > 1 : # in case that only remain
            # suppres those too close (r diff < r_supper) and those too close to the left/right-most (r diff < r_supper)
            n_angle = [remain for i, remain in enumerate(angle) if (r[i] - r[i-1] > r_supper) and abs(r[i] - maxmin_r[0]) > r_supper and abs(r[i] - maxmin_r[1]) > r_supper]
            n_r = [remain for i, remain in enumerate(r) if (r[i] - r[i-1] > r_supper) and abs(r[i] - maxmin_r[0]) > r_supper and abs(r[i] - maxmin_r[1]) > r_supper]
        else:
            n_angle = angle
            n_r = r
        # add the left/right-most back
        n_angle += maxmin_angle
        n_r += maxmin_r
        # sorted again
        n_angle = [remain for _, remain in sorted(zip(n_r,n_angle))]
        n_r.sort()
        
        # filter out some pixel that at 255 with 254
        img = np.where(img == 255, 254, img)
        if plot == True:
            img_keep2 = img.copy()
        # check the line from left to right
        left_img_all = []
        right_img_all = []
        for i in range(0, len(n_angle)):
            rho = n_r[i]
            theta = n_angle[i]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            img_keep = img.copy()
            img_line = cv2.line(img_keep, pt1, pt2, (0,255,0), 1)
            if plot == True:
                img_line_out = cv2.line(img_keep2, pt1, pt2, (0,255,0), 1)

            # get the index of location green line
            h, w = np.where(img_line[...,1] == 255)
            left_img = np.empty((h.shape[0],0,3), dtype=int)
            right_img = np.empty((h.shape[0],0,3), dtype=int)

            # create the close edage ndarray
            # concatenate along w direction
            for j in range(1,6):
                left = img_line[h,w-1-j,:] # the 1 is for safety only
                right = img_line[h,w+1+j,:]
                left = np.expand_dims(left, axis=1)
                right = np.expand_dims(right, axis=1)
                left_img = np.concatenate((left, left_img), axis=1)
                right_img = np.concatenate((right_img,right), axis=1)
            left_img_all.append(left_img)
            right_img_all.append(right_img)
            # image = np.concatenate((left_img, right_img), axis=1)
            # fig, axs = plt.subplots()
            # axs.imshow(image.astype(np.uint8))
            # fig.savefig(f'EDA/edage_image_{i}.png')
    else:
        raise IOError('No line detected')
    fig, ax = plt.subplots()
    fig.set_figheight(15)
    fig.set_figwidth(15)
    ax.imshow(img_line_out.astype(np.uint8))
    plt.savefig('EDA/hough_all.png')

    return angle_all, r_all, n_angle, n_r, left_img_all, right_img_all

def curve_detect(left_img_list, right_img_list, plot=False):
    righty = []
    median_diff_list = []
    median_left_list = []
    median_right_list = []
    for i in range(len(left_img_list)):
        median_left = []
        median_right = []
        righty_left = []
        righty_right = []
        median_diff = []
        for j, _ in enumerate(left_img_list[i]):
            median_left.append(np.median(left_img_list[i][j]))
            median_right.append(np.median(right_img_list[i][j]))
            median_diff.append(median_left[j] - median_right[j])

            # only see the lower part of line, since upper part might curve
            if j > 200:
                righty_left.append(np.median(left_img_list[i][j]))
                righty_right.append(np.median(right_img_list[i][j]))

        median_diff_list.append(median_diff)
        median_left_list.append(median_left)
        median_right_list.append(median_right)
        if plot == True:
            fig, axs = plt.subplots()
            axs.plot(median_left, label='left side')
            axs.plot(median_right, label='right side')
            axs.legend()
            axs.set_title('Edage_median')
            fig.savefig(f'EDA/Edage_median_{i}.png')

        # check the line is on the rightside of metal or not            
        if np.median(righty_left) > np.median(righty_right):
            righty.append(True)
        else:
            righty.append(False)
    
    return righty, median_diff_list, median_left_list, median_right_list

if __name__ == "__main__":
    start_time = time.time()
    img_back = np.load('data/background.npy')
    img_target = 'data/clean_data/U100/1500_curve_3/frame0570.jpg'
    img_residual, img_edage, img_target_ori = residual(img_target, img_back)

    threshold = 100
    r_supper = 10

    angle, r, n_angle, n_r, left_img_all, right_img_all = hough(img_edage, 1, np.pi/180, threshold, img_target_ori, r_supper=r_supper, plot=True)
    print('\nall angle: ', angle)
    print('all r: ',r)
    print('suppress angle: ', n_angle)
    print('suppress r: ',n_r)
    print(f'\n--- hough spends: {time.time() - start_time:.2f} sec ---\n')

    righty, diff_list, left_list, right_list = curve_detect(left_img_all, right_img_all)
    print('righty:', righty)
    print(f'\n--- curve_detect spends: {time.time() - start_time:.2f} sec ---\n')

    for i, diff in enumerate(diff_list):
        fig, axs = plt.subplots(2,1, sharex=True)
        axs[0].plot(diff)
        axs[0].set_title(f'Edage_median_diff_{i}')
        axs[1].plot(left_list[i], label='left side')
        axs[1].plot(right_list[i], label='right side')
        axs[1].legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        axs[1].set_title(f'Edage_median_{i}')
        fig.savefig(f'EDA/Edage_median_diff_{i}.png')

    ## save backround
    # image_dir = '/home/rico-li/Job/豐興鋼鐵/data/U100/1740_curve_3'
    # image_paths = os.listdir(image_dir)
    # image_paths.sort()
    # image_paths = image_paths[:199] # 199 images as average background
    # image_paths = [os.path.join(image_dir, image_path) for image_path in image_paths]
    # img_list = image_crop(image_paths)
    # img_back = image_back(img_list)
    # np.save('data/background.npy', img_back)