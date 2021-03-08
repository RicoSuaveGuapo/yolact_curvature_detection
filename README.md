# Overall Process and Structure

## Information
### Data
* RGB of the dataset:
   mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
### Structure
#### Useful `.py`
* `labelme2coco.py`: for transfering the labelme annotation files into yolact format (coco). Note that I had modified some parts which is different from original one in labelme.
* `video2image.py`: for transfering video to images, note that I had put the multiprocessing in it. 
#### Folder
* data: save all the data
* prediction_result: the results so far
* temp_result: the tempary results, to see the final result check prediction_result folder.
* yolact: the instance segmentation model and data.

#### (Decrpyted)
* `image_capture.py`: for checking the whether metal entering. Since yolact can detect the image itself, may not need this anymore.
   * `image_capture.py` can also distinguish U150/U100 and A30.
* `tranditional_main.py`: using traditional method to detect the curve.
* `traditional_method.ipynb`: same as `tranditional_main.py`

## Traditional Method (Decrypted)
1. Line choosing
   1. Limit the number of detecting
   2. require that left pixel value must be greater than right pixel
2. Curve detection
   1. gradient (V)
   2. value (V)
May not gonna work

## Semantic Segmentation
### 1. Data Preparing
1. Data format: label data is `.png` file (V)
2. how to convert `json_to_dataset` to multiple images dataset (V)
3. data spilt (V)
4. workflow of human labeling
   1. Two classes, curve and ok (V)
   2. each 50 images (V)
   3. Preparing Lecture (V)
5. people (V)
6. Contacting Mr. 洪 for the data (V)

### 2. Sementic Segmentation Model (Decrypted)
"1. The pixel values in annotations `.png` three channels are exactly the same (V)
2. Understand how data is read-in (V)
3. Crop the input data to smaller size (V)
4. Need to under stand object150_info.csv and color150.mat (V)
   1. color150.mat is solely for visualization
   2. object150_info.csv is for test.py only (for class IoU)
5. Understand how accuracy work (V)
6. Understand how the model know how many classes (V)
7. Loaded in pretrained model
   1. pretrained models are save in here http://sceneparsing.csail.mit.edu/model/pytorch/. Modified `demo_test.sh` to download other pretrained model `.pth`
   2. If the `start_epoch` in the config is changed to > 0, then the model pretrained will be loaded. 
8. Background substraction
9. Which model suitable?
10. Change the label of the background (V)
11. Crop the output image

### 3. Instance Segmentation Model
1. data read in (V)
2. Check the segmentation (V)

#### 3.1 Result checking
1. Get the "accuracy" with fixing IoU (V)
2. Metric: Above 0.5 IoU, and with any confidence curve kind recall (V)

#### 3.2 Improvement
1. change the weight of cls/bbox/mask? More weight on bbox or class? change bbox weight from 1.5 to 3
   1. Now change to 6 (V)
2. Use normal NMS, not FasNMS (V) good for U100_ok, 
   1. use for all the inference
3. Change the thresold of NMS (V) not good
   1. Not adopted
4. Use YOLACT++ model (V), remember use torch 1.4 and torchvision 0.5 see here (https://github.com/dbolya/yolact/issues/431#issuecomment-622651477) for using YOLACT++ model
5. use the `preserve_aspect_ratio` in `config.py` for improve input resolution
   1. formers are 550*550
   2. original images are 410*700


### 4. GAN on Instance Segmentation Model (Optional)
1. Try using the GAN (try with normal YOLACT first)
   1. Understand the YOLACT detail model
      1. Check the backbone in `backbone.py` (V)
      2. Check the `FPN` (V)
      3. Check Protonet in `functions.py` `make_net` (V)
      4. Prediction Head (V)
      5. Prediction Mask Output (V)
      6. Original Image & Ground Truth Map (V)
      7. check `def lincomb_mask_loss` for how to get lincomb output (V)
      8. check `output_utils.py` for correct outoput (V)
   2. Connect a simple GAN onto it
      1. Simple Discriminator (V)
      2. Prediction map format options (bilinear mode)
         1. align_corners True
         2. align_corners False (V)
      3. Size options
         1. downsample the gt mask and image (YOLACT choice, GAN on Seg also used it) (V)
         2. upsample the prediction map
      4. Connect to the original model (V)
   3. Train it, and compared with before
      1. Grid Search on lambda (Tune Ricky adivce https://docs.ray.io/en/master/tune/index.html)
      2. Slowing alternating scheme (500 iterations change Gen <-> Dis) (V)
      3. Different learning rates, smaller for discriminator (V)
      4. Need to check `eval.py` (V)
      5. Modified the discriminator optimizer detail structure (-)
      6. Modified the discriminator structure
         1. Try to follow the https://arxiv.org/abs/1611.08408 Stanford background dataset first (V)
         2. Try to follow DCGAN
            1. add the batch_norm2D and change the inital weight (V)
            2. Modified the training process (V)
         3. Try to follow WGAN
            1. Implemet WLoss (V)
         4. Try GAN Mask R-CNN (2010.13757)
            1. Using their discrimantor structure (V)
            2. Output featmap from discriminator, using L1 distance. (V)
            3. Adopted their schedular?
         5. Try GANHack
      7. Push on server (-)
         1. Check the Netloss and CustomDataParallel is correct. (V)
         2. Using float16 (V)
            1. Modify the mask loss from BCE to smooth_l1_loss, since the BCE 
            is not safe for autocast
            2. Modify the `tra_gan_server.py` for `@autocast()` before NetLoss class forward.
         3. Optimize training speed
            1. Change the container (CPU 40 cores) (V)
            2. Use DistributedDataParallel 

### 5. Other Instance Segmentation Models
1. Using SOLOv2 or SipMask.
