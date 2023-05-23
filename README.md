# Slovo - Russian Sign Language Dataset

We introduce a large-scale video dataset **Slovo** for Russian Sign Language task. Slovo dataset size is about **16 GB**, and it contains **20400** RGB videos for **1000** sign language gestures from 194 singers. Each class has 20 samples. The dataset is divided into training set and test set by subject `user_id`. The training set includes 15300 videos, and the test set includes 5100 videos. The total video recording time is ~9.2 hours. About 35% of the videos are recorded in HD format, and 65% of the videos are in FullHD resolution. The average video length with gesture is 50 frames.

For more information see our paper - [arXiv]().

## Downloads
### [Main download link](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/slovo/slovo.zip)

|                                                                                               Downloads | Size (GB) | Comment                                                              |
|--------------------------------------------------------------------------------------------------------:|:----------|:---------------------------------------------------------------------|
|                [Slovo](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/slovo/slovo.zip) | ~16       | Trimmed HD+ videos by `(start, end)` annotations                     |
|          [Origin](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/slovo/slovo_full.zip) | ~105      | Original HD+ videos from mining stage                                |
|        [360p](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/slovo/slovo_full360.zip)  | ~13       | Resized original videos by `min_side = 360`                          |
| [Landmarks](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/slovo/slovo_mediapipe.json) | ~1.2      | Mediapipe hand landmark annotations for each frame of trimmed videos |

Also, you can download **Slovo** from [Kaggle](https://www.kaggle.com/datasets/kapitanov/slovo).

Annotation file is easy to use and contains some useful columns, see `annotations.csv` file:

|    | attachment_id | user_id | width | height | length |  text  | train   | begin | end |
|---:|:--------------|:--------|------:|-------:|-------:|-------:|:--------|:------|:----|
|  0 | de81cc1c-...  | 1b...   |  1440 |   1920 |     14 | привет | True    | 30    | 45  |
|  1 | 3c0cec5a-...  | 64...   |  1440 |   1920 |     32 |   утро | False   | 43    | 66  |
|  2 | d17ca986-...  | cf...   |  1920 |   1080 |     44 |  улица | False   | 12    | 31  |

where:
- `attachment_id` - video file name
- `user_id` - unique anonymized user ID
- `width` - video width
- `height` - video height
- `length` - video length
- `text` - gesture class in Russian Langauge
- `train` - train or test boolean flag
- `begin` - start of the gesture (for original dataset)
- `end` - end of the gesture (for original dataset)

For convenience, we have also prepared a compressed version of the dataset, in which all videos are processed by the minimum side `min_side = 360`. Download link - **[slovo360p](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/slovo/slovo_full360.zip)**.
Also, we annotate trimmed videos by using **MediaPipe** and provide hand keypoints in [this annotation file](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/slovo/slovo_mediapipe.json).

## Models
We provide some pre-trained models as the baseline for Russian sign language recognition. 
We tested models with frames number from [16, 32, 48], and **the best for each are below**.
The first number in the model name is frames number and the second is frame interval. 

| Model Name        | Model Size (MB) | Metric | ONNX                                                                                                            | TorchScript                                                                                                 |
|-------------------|-----------------|--------|-----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| MViTv2-small-16-4 | 140.51          | 58.35  | [weights](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/slovo/models/mvit/onnx/mvit16-4.onnx) | [weights](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/slovo/models/mvit/pt/mvit16-4.pt) |
| MViTv2-small-32-2 | 140.79          | 64.09  | [weights](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/slovo/models/mvit/onnx/mvit32-2.onnx) | [weights](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/slovo/models/mvit/pt/mvit32-2.pt) |
| MViTv2-small-48-2 | 141.05          | 62.18  | [weights](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/slovo/models/mvit/onnx/mvit48-2.onnx) | [weights](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/slovo/models/mvit/pt/mvit48-2.pt) |
| Swin-large-16-3   | 821.65          | 48.04  | [weights](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/slovo/models/swin/onnx/swin16-3.onnx) | [weights](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/slovo/models/swin/pt/swin16-3.pt) |
| Swin-large-32-2   | 821.74          | 54.84  | [weights](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/slovo/models/swin/onnx/swin32-2.onnx) | [weights](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/slovo/models/swin/pt/swin32-2.pt) |
| Swin-large-48-1   | 821.78          | 55.66  | [weights](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/slovo/models/swin/onnx/swin48-1.onnx) | [weights](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/slovo/models/swin/pt/swin48-1.pt) |
| ResNet-i3d-16-3   | 146.43          | 32.86  | [weights](https://sc.link/jRBRY) | [weights](https://sc.link/lY0Y7) |
| ResNet-i3d-32-2   | 146.43          | 38.38  | [weights](https://sc.link/kRgRr) | [weights](https://sc.link/mZkZE) |
| ResNet-i3d-48-1   | 146.43          | 43.91  | [weights](https://sc.link/gJyJZ) | [weights](https://sc.link/n5l5Y) |

## Demo
```console
usage: demo.py [-h] -p CONFIG [--mp] [-v] [-l LENGTH]

optional arguments:
  -h, --help            show this help message and exit
  -p CONFIG, --config CONFIG
                        Path to config
  --mp                  Enable multiprocessing
  -v, --verbose         Enable logging
  -l LENGTH, --length LENGTH
                        Deque length for predictions


python demo.py -p <PATH_TO_CONFIG>
```

## Authors and Credits
- [Kapitanov Alexander](https://www.linkedin.com/in/hukenovs)
- [Kvanchiani Karina](https://www.linkedin.com/in/kvanchiani)
- [Nagaev Alexander](https://www.linkedin.com/in/nagadit/)
- [Petrova Elizaveta](https://www.linkedin.com/in/elizaveta-petrova-248135263/)

## Citation
You can cite the paper using the following BibTeX entry:

    @article{Slovo,
        title={Slovo - Russian Sign Language Dataset and Models},
        author={Kapitanov, Alexander and Kvanchiani, Karina and Nagaev, Alexander and Petrova, Elizaveta},
        journal={arXiv preprint <link>},
        year={2023}
    }

## Links
- [arXiv]()
- [Kaggle](https://www.kaggle.com/datasets/kapitanov/slovo)
- [Habr](https://habr.com/ru/companies/sberdevices/articles/737018/)
- [Github](https://github.com/hukenovs/slovo)
- [Gitlab](https://gitlab.aicloud.sbercloud.ru/rndcv/slovo)
- [Paperswithcode]()

## License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a variant of <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

Please see the specific [license](https://github.com/hukenovs/slovo/blob/master/license/en_us.pdf).
