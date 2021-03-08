# Data

## Information
### Folder
* clean_data_20frames: the main target data
* dirty_data: contains person or light change images
* label_inistruction: the labels provided by the company
* original_data: the raw image/videos data
### File
* `background.npy`: the average value of the dataset (saved in 2D numpy array)
* `data_summary.py`: the simple script to summarize the data from label_inistruction
* `labels.txt`: example of label class which will be used in yolact.


## Data Summary
* All video count
    |根數|正常|彎曲|稍微彎曲|
    |:-:|:-:|:-:|:-:|
    |3根|2|5|1|
    |4根|1|1|0|
    |5根|2|0|1|
    |13根|1|2|0|
    |15根|1|1|0|
    |Total|7|9|2|
* U150
    1621 curve 3: 6 sets
    1634 curve 3: 6 sets
    1645 ok 3: 6 sets

    |根數|正常|彎曲|稍微彎曲|
    |:-:|:-:|:-:|:-:|
    |3根|1|1|0|
* U100 (sets: how many targets)
    1500 curve 3: 6 sets
    1505 curve 3: 8 sets
    1518 ok 3: 9 sets
    1525 curve 4: 9 sets
    1528 minor curve 5
        images_1: 1 set
        images_2: 6 sets
    1534 ok 5: 8 sets
    1740 curve 5: 6 sets
    1745 small surve 4: 6 sets
    1750 ok 4: 9 sets
    1755 ok 5: 9 sets

    |根數|正常|彎曲|稍微彎曲|
    |:-:|:-:|:-:|:-:|
    |3根|1|2|1|
    |4根|1|1|0|
    |5根|2|1|1|
* A30
    1555 curve 13: 16 sets
    1609 curve 13: 15 sets
    1613 ok 13: 15 sets
    1620 curve 15: 15 sets
    1633 ok 15: 16 sets

    |根數|正常|彎曲|稍微彎曲|
    |:-:|:-:|:-:|:-:|
    |13根|1|2|0|
    |15根|1|1|0|
