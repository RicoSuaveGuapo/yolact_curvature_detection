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
* 