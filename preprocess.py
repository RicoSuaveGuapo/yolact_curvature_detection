import os
import cv2
import math
import glob
import json
import time
import numpy as np
import matplotlib.pyplot as plt

def image_crop(image_paths):
    if type(image_paths) == list:
        img_list = []
        for image_path in image_paths:
            img = cv2.imread(image_path)
            img = img[70:, :700, :]
            img_list.append(img)
        return img_list
    
    elif type(image_paths) == str:
        img = cv2.imread(image_paths)
        img = img[70:, :700, :]
        return img
    
    elif type(image_paths) == np.ndarray:
        img = image_paths
        img = img[70:, :700, :]
        return img
    else:
        raise TypeError('Not supportted data type')

def image_back(img_list:list):
    '''
    1 sec (25 images) will be enough, 
    '''
    h, w, c = img_list[0].shape
    img_back = np.empty((h, w, c), int)
    for img in img_list:
        img_back += img
    img_back = img_back/len(img_list)
    img_back = img_back.astype(np.float32)
    return img_back

def residual(img_path, img_back, canny_min=100, canny_max=200):
    img = image_crop(img_path)
    res = np.abs(img - img_back)
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, np.ones((2,2))) # denoising
    img_edage = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    img_edage = cv2.Canny(img_edage, canny_min, canny_max)
    return res, img_edage, img

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def data_split(labels:list, images:list)->list:
    np.random.seed(42)
    kinds = np.unique(labels)

    train_labels = []
    train_images = []
    val_labels = []
    val_images = []
    test_labels = []
    test_images = []
    for kind in kinds:
        kind_label = [label for label in labels if label == kind]
        kind_image = [images[i] for i, label in enumerate(labels) if label == kind]

        index = np.random.permutation(range(0,len(kind_label)))
        # kind_label = kind_label[index] # already in the same label
        kind_image = [kind_image[i] for i in index]

        train_labels += kind_label[:int(len(kind_label)*3/7)]
        val_labels += kind_label[int(len(kind_label)*3/7):int(len(kind_label)*5/7)]
        test_labels += kind_label[int(len(kind_label)*5/7):]

        train_images += kind_image[:int(len(kind_image)*3/7)]
        val_images += kind_image[int(len(kind_image)*3/7):int(len(kind_image)*5/7)]
        test_images += kind_image[int(len(kind_image)*5/7):]

    return (train_labels, train_images), (val_labels, val_images), (test_labels, test_images)


def modified_json(json_path, mode, kind):
    """
    Crop the json to desire size.
    --- Notice ---
    Note that if an annotation has too many label points above the crop region
    it will be discard!
    """
    json_files = os.listdir(json_path)
    json_files = [os.path.join(json_path, json_file)  for json_file in json_files if json_file.endswith('.json')]

    for json_file in json_files:
        with open(json_file,"r+") as f:
            data = json.load(f)
        shapes = data['shapes'] # shapes [{'label': 'U100_curve', 'points': [[357..}, ]
        new_shapes = []

        outsider = 0
        for shape in shapes:
            points = shape['points']
            new_points = [[point[0],point[1]-70] for point in points]
            y = [point[1]-70 for point in points]
            outsider += sum(y)
            shape['points'] = new_points
            new_shapes.append(shape)

        json_name = os.path.basename(json_file)
        if outsider > 0:
            data['shapes'] = new_shapes
            path = f'/home/rico-li/Job/豐興鋼鐵/data/clean_data_20frames/{kind}/annotations/{mode}/'
            if not os.path.exists(path):
                os.mkdir(path)
            with open(f"{path}mod_{json_name}", "w") as k:
                json.dump(data, k)
        
        elif outsider < 0:
            # discard the one annotation mostly outside of the cropped region
            print(f'{json_name} labels mostly located outside of the cropped region')
            pass


def create_coco_json(mod_json_path, des_path, labels_path, mode=None, kind=None, mod=True, ori_json_path=None):
    if mod:
        modified_json(ori_json_path, mode, kind)
        if not os.path.exists(mod_json_path):
            os.mkdir(mod_json_path)
        os.system(f'/home/rico-li/Job/豐興鋼鐵/labelme2coco.py {mod_json_path} {des_path} --labels {labels_path}')
    else:
        os.system(f'/home/rico-li/Job/豐興鋼鐵/labelme2coco.py {mod_json_path} {des_path} --labels {labels_path}')

def json_check(json_path, mode, correct_label:list, remove_dup=True, rm_wrong=False):
    json_files = os.listdir(json_path)
    json_files = [os.path.join(json_path, json_file)  for json_file in json_files if json_file.endswith('.json')]
    json_names = [os.path.basename(json_file) for json_file in json_files]

    # remove duplicate
    if remove_dup:
        dup_json = [dup for dup in json_files if  dup.find('(1)') != -1]
        print(f'there are {len(dup_json)} duplicate json')
        for dup in dup_json:
            os.remove(dup)
        if len(dup_json) == 0:
            print('Already removed the duplicate')
        else:
            print('Removed')

    wrong_name = []
    for i, json_file in enumerate(json_files):
        with open(json_file,"r+") as f:
            data = json.load(f)
        shapes = data['shapes'] # shapes [{'label': 'U100_curve', 'points': [[357..}, ]
        for shape in shapes:
            label = shape['label']
            if (label != correct_label[0]) and (label != correct_label[1]):
                print(f'wrong label at {json_names[i]}')
                print(f'Has label: {label}')
                wrong_name.append(json_names[i])
    print(f'All {mode} Check')

    if not rm_wrong:
        return wrong_name

    elif rm_wrong:
        wrong_name = [os.path.join(json_path,w_name) for w_name in wrong_name]
        for w_name in wrong_name:
            os.remove(w_name)
        return print('removed wrong label')
    

def images2video(image_dir_path, output_name, fps, kind):
    img_array = []
    filenames = os.listdir(image_dir_path)
    filenames.sort()
    filenames = [os.path.join(image_dir_path,filename) for filename in filenames if (filename.endswith('.jpg')) or (filename.endswith('.png'))]
    for filename in filenames:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    out = cv2.VideoWriter(f'./Prediction/{kind}/{output_name}.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()




if __name__ == "__main__":
    kind = 'U150'
    modes = ['train','val']
    # Step of downloading json label files
    # 1. Download the json files to /home/rico-li/Job/豐興鋼鐵/data/clean_data_20frames/U100/images
    # 2. check whether there is wrong labels, delect those wrong labels
    
    # json_check(f'/home/rico-li/Job/豐興鋼鐵/data/clean_data_20frames/{kind}/images/final_train/train', mode='train', correct_label=[f'{kind}_ok',f'{kind}_curve'], rm_wrong=True)
    # json_check(f'/home/rico-li/Job/豐興鋼鐵/data/clean_data_20frames/{kind}/images/final_val/val',mode='val', correct_label=[f'{kind}_ok',f'{kind}_curve'], rm_wrong=True)
    # 3. Creating cropped json and images, and also create coco_style json
    # remeber to put the original json file under the /images/mode
    # create mod_json first
    # for mode in modes:
    #     create_coco_json(ori_json_path=f'/home/rico-li/Job/豐興鋼鐵/data/clean_data_20frames/{kind}/images/final_{mode}/{mode}', 
    #                 mod_json_path=f'/home/rico-li/Job/豐興鋼鐵/data/clean_data_20frames/{kind}/annotations/{mode}',
    #                 des_path=f'/home/rico-li/Job/豐興鋼鐵/data/clean_data_20frames/{kind}/annotations/yolact_{mode}',
    #                 labels_path='/home/rico-li/Job/豐興鋼鐵/data/labels.txt', mode=mode, kind=kind)

    # move the json to data/clean_data_20frames/mode
    # create all second
    # for mode in modes:
    #     create_coco_json(mod_json_path=f'data/clean_data_20frames/{mode}',
    #                 des_path=f'/home/rico-li/Job/豐興鋼鐵/data/clean_data_20frames/yolact_{mode}',
    #                 labels_path='/home/rico-li/Job/豐興鋼鐵/data/labels.txt', mod=False)


    # create result video
    fps = 10
    pred_path = '/home/rico-li/Job/豐興鋼鐵/Prediction/U150/images'
    images2video(image_dir_path=pred_path, output_name='U150', fps=fps, kind='U150')
    
    # path = '/home/rico-li/Job/豐興鋼鐵'
    # create_coco_json(path, 
    #                 '/home/rico-li/Job/豐興鋼鐵/temp_json',
    #                 '/home/rico-li/Job/豐興鋼鐵/data/labels.txt')

    # a = [0]*7+[1]*7
    # b = [i for i in range(15)]
    # print(data_split(a,b))

    # lines = cv2.HoughLines(img_edage,1,np.pi/180,120)
    # dst: edge image. It should be a grayscale image
    # lines: A vector store (r,θ) of the detected lines
    # r is the perpendicular distance from origin to the line
    # θ is the angle formed by this perpendicular line and horizontal axis measured
    # in counter-clockwise 
    # (That direction varies on how you represent the coordinate system. 
    # This representation is used in OpenCV)
    # rho : The resolution of the parameter r in pixels. We use 1 pixel.
    # theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
    # threshold: The minimum number of intersections to "*detect*" a line

    # img = img_target_ori.copy()
    # if lines is not None:
    #     lines = lines.squeeze()
    #     angle = [line[-1] for line in lines if (line[-1] < 30 * np.pi/180) and (line[0] > 200)]
    #     r = [line[0] for line in lines if (line[-1] < 30 * np.pi/180) and (line[0] > 200)]
    #     for i in range(0, len(angle)):
    #         rho = r[i]#lines[i][0][0]
    #         theta = angle[i]#lines[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    #         img_line = cv2.line(img, pt1, pt2, (0,255,0), 1)
    # else:
    #     raise IOError('No line detected')

    # fig, axs = plt.subplots(2,3, sharey=True, sharex=True)
    # fig.set_figheight(15)
    # fig.set_figwidth(15)
    # axs[0,0].imshow(img_back.astype(np.uint8))
    # axs[0,0].set_title('background')
    # axs[0,1].imshow(img_target_ori)
    # axs[0,1].set_title('Target')
    # axs[1,0].imshow(img_residual.astype(np.uint8))
    # axs[1,0].set_title('Residual after morphologyEx')
    # axs[1,1].imshow(img_edage)
    # axs[1,1].set_title('Residual with Canny')
    # axs[1,2].imshow(img_line.astype(np.uint8))
    # axs[1,2].set_title('Hough')
    # plt.show()

    # fig, ax = plt.subplots(1,1)
    # fig.set_figheight(15)
    # fig.set_figwidth(15)
    # ax.imshow(img_line.astype(np.uint8))
    # plt.show()

    # --- test image pixel difference ---
    # img_target = image_crop(img_target)
    # res_line = img_target[-1,...] - img_back[-1,...]
    # print('Metal entering')
    # print(res_line.shape)
    # print(np.median(res_line))
    # 4.562813

    # --- check the threshold of variance of pixel in background --- 
    # image_dir = 'data/20200817_A30x4t_1620_彎曲_15支'
    # image_paths = os.listdir(image_dir)
    # image_paths.sort()
    # image_paths = image_paths[:500]
    # image_paths = [os.path.join(image_dir, image_path) for image_path in image_paths]
    # img_list = image_crop(image_paths[:25])
    # img_back = image_back(img_list)
    # img_median = 0
    # for image_path in image_paths:
    #     first_back = image_path
    #     first_back = image_crop(first_back)
    #     first_res_line = first_back[-1,...] - img_back[-1,...]
    #     img_median += np.median(first_res_line)
    # print(img_median/len(image_paths))
    pass