# Main Target Dataset

## Folder
The file name with 'mod' in the begining is cropped from 944x480 -> 700x410 (cropped to lower left). The annotations are the same.

* In A30/U100/U150, each one contains annotations and original_images folders. 
    * annotations: contains
        * train/val: voc format annotation
        * yolact_train/_val: coco format annotation
    * original_images: the orignial images (ignorer the ok/curve in the file name). The ok/curve is detail listed in "data/label_instruction" folder.
* In all_train/val_annotation: contains all the annotations in voc format.
* yolact_train/_val: contains all the annotations for yolact to eat.