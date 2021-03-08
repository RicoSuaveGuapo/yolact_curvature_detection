# YOLACT 
Only the custom file will be documented in here. To use YOLACT, please check its original [README.md](https://github.com/dbolya/yolact/blob/master/README.md) in github.

## Folder
* GAN_Mask_R-CNN (Decrypted): implementation of GAN on segmentation
* reserve_weight: saving the model weight in it. Check 'reserve_weight/README.md' for detail.
* data: for coco images and annotations. NOT for your dataset!
* temp_result: some in-process temporary results.
* weights: You can use the `darknet53.pth` and `resnet101_reducedfc.pth` here. Don't need to download again. I have forgot the target for other weights... Check reserve_weight instead.
* result: the final result after you run eval.py in quantitative. Map and bbox information can be found here.

## File
* `criterion_dis.py`: GAN losses for different structures.
* `result_eval.py`: Output the AP curve, takes the content of 'yolact/results' as input.
* `train.py`: original training routine
* `train_gan.py`: GAN added training routine
* `train_gan_server.py`: GAN added training routine applied to server
* `output.py`: a simple script including training and evaluation. (Note that the path has to changed depending on what you need)