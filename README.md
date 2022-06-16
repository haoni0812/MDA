## Meta Distribution Alignment for Generalizable Person Re-Identification (MDA), [[CVPR 2022](https://arxiv.org/abs/2011.14670)]


---

## 1) Prerequisites

- Ubuntu 18.04
- Python 3.6
- Pytorch 1.7+
- NVIDIA GPU (>=8,000MiB)
- Anaconda 4.8.3
- CUDA 10.1 (optional)
- Recent GPU driver (Need to support AMP [[link](https://pytorch.org/docs/stable/amp.html)])




```

## 2) download dataset and connect it

- Download dataset
  - For single-source DG
    - Need to download Market1501, DukeMTMC-REID [check section 8-1,2]
  - For multi-source DG
    - Training: Market1501, DukeMTMC-REID, CUHK02, CUHK03, CUHK-SYSU [check section 8-1,2,3,4,5]
    - Testing: GRID, PRID, QMUL i-LIDS, VIPer [check section 8-6,7,8,9]

- Symbolic link (recommended)
  - Check `symbolic_link_dataset.sh`
  - Modify each directory (need to change)
  - `cd MetaBIN`
  - `bash symbolic_link_dataset.sh`
  
- Direct connect (not recommended)
  - If you don't want to make symbolic link, move each dataset folder into `./datasets/`
  - Check the folder name for each dataset

## 3) Create pretrained and logs folder

- Symbolic link (recommended)
  - Make 'MetaBIN(logs)' and 'MetaBIN(pretrained)' folder outside MetaBIN
```
├── MetaBIN
│   ├── configs/
│   ├── ....
│   ├── tools/
├── MetaBIN(logs)
├── MetaBIN(pretrained)
```
  - `cd MetaBIN`
  - `bash symbolic_link_others.sh`
  - Download pretrained models and change name 
    - mobilenetv2_x1_0: [[link](https://mega.nz/#!NKp2wAIA!1NH1pbNzY_M2hVk_hdsxNM1NUOWvvGPHhaNr-fASF6c)]
    - mobilenetv2_x1_4: [[link](https://mega.nz/#!RGhgEIwS!xN2s2ZdyqI6vQ3EwgmRXLEW3khr9tpXg96G9SUJugGk)]
    - change name as `mobilenetv2_1.0.pth`, `mobilenetv2_1.4.pth`
  - Or download pretrained models [[link](https://drive.google.com/u/0/uc?id=1o-MqjM1YBeUoZB5mNlGiB6LVSIA2RV71&export=download)]

- Direct connect (not recommended)
  - Make 'pretrained' and 'logs' folder in `MetaBIN`
  - Move the pretrained models to `pretrained`
  

## 7) Train

- If you run code in pycharm
  - tools/train_net.py -> Edit congifuration
  - Working directory: `your folders/MetaBIN/`
  - Parameters: `--config-file ./configs/Sample/DG-mobilenet.yml`

- Single GPU

`python3 ./tools/train_net.py --config-file ./configs/Sample/DG-mobilenet.yml`

- Single GPU (specific GPU)

`python3 ./tools/train_net.py --config-file ./configs/Sample/DG-mobilenet.yml MODEL.DEVICE "cuda:0"`

- Resume (model weights is automatically loaded based on `last_checkpoint` file in logs)

`python3 ./tools/train_net.py --config-file ./configs/Sample/DG-mobilenet.yml --resume`

- Evaluation only

`python3 ./tools/train_net.py --config-file ./configs/Sample/DG-mobilenet.yml --eval-only`


