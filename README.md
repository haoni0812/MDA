## Meta Distribution Alignment for Generalizable Person Re-Identification (MDA)


---

## 1) Prerequisites

- Ubuntu 18.04
- Python 3.6
- Pytorch 1.7+
- NVIDIA GPU (>=8,000MiB)
- Anaconda 4.8.3
- CUDA 10.1 (optional)
- Recent GPU driver (Need to support AMP [[link](https://pytorch.org/docs/stable/amp.html)])


## 2) Datasets

- (1) Market1501
  - Create a directory named `Market-1501-v15.09.15`
  - Download the dataset to `Market-1501-v15.09.15` from [link](http://www.liangzheng.org/Project/project_reid.html) and extract the files.
  - The data structure should look like
  ```
  Market-1501-v15.09.15/
  ├── bounding_box_test/
  ├── bounding_box_train/
  ├── gt_bbox/
  ├── gt_query/
  ├── query/
  ```

- (2) DukeMTMC-reID
  - Create a directory called `DukeMTMC-reID`
  - Download `DukeMTMC-reID` from [link](http://vision.cs.duke.edu/DukeMTMC/) and extract the files.
  - The data structure should look like
  ```
  DukeMTMC-reID/
  ├── bounding_box_test/
  ├── bounding_box_train/
  ├── query/
  ```

- (3) CUHK02
  - Create `cuhk02` folder
  - Download the data from [link](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html) and put it under `cuhk02`.
    - The data structure should look like
  ```
  cuhk02/
  ├── P1/
  ├── P2/
  ├── P3/
  ├── P4/
  ├── P5/
  ```
  
- (4) CUHK03
  - Create `cuhk03` folder
  - Download dataset to `cuhk03` from [link](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html) and extract “cuhk03_release.zip”, resulting in “cuhk03/cuhk03_release/”.
  - Download the new split (767/700) from person-re-ranking. What you need are “cuhk03_new_protocol_config_detected.mat” and “cuhk03_new_protocol_config_labeled.mat”. Put these two mat files under `cuhk03`.
  - The data structure should look like
  ```
  cuhk03/
  ├── cuhk03_release/
  ├── cuhk03_new_protocol_config_detected.mat
  ├── cuhk03_new_protocol_config_labeled.mat
  ```
  
- (5) Person Search (CUHK-SYSU)
  - Create a directory called `CUHK-SYSU`
  - Download `CUHK-SYSU` from [link](https://github.com/ShuangLI59/person_search) and extract the files.
  - Cropped images can be created by my matlab code `make_cropped_image.m` (this code is included in the datasets folder)
  - The data structure should look like
  ```
  CUHK-SYSU/
  ├── annotation/
  ├── Image/
  ├── cropped_image/
  ├── make_cropped_image.m (my matlab code)
  ```


- (6) GRID
  - Create a directory called `GRID`
  - Download `GRID` from [link](http://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/underground_reid.zip) and extract the files.
  - Split sets (`splits.json`) can be created by python code `grid.py`
  - The data structure should look like

  ```
  GRID/
  ├── gallery/
  ├── probe/
  ├── splits_single_shot.json (This will be created by `grid.py` in `fastreid/data/datasets/` folder)
  ```

  
- (7) PRID
  - Create a directory called `prid_2011`
  - Download `prid_2011` from [link](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/) and extract the files.
  - Split sets (`splits_single_shot.json`) can be created by python code `prid.py`
  - The data structure should look like

  ```
  prid_2011/
  ├── single_shot/
  ├── multi_shot/
  ├── splits_single_shot.json (This will be created by `prid.py` in `fastreid/data/datasets/` folder)
  ```
  
  
- (8) QMUL i-LIDS
  - http://www.eecs.qmul.ac.uk/~jason/data/i-LIDS_Pedestrian.tgz
  - https://github.com/BJTUJia/person_reID_DualNorm
  - Create a directory called `QMUL_iLIDS`
  - Download `QMUL_iLIDS` from the upper links
  - Split sets can be created by python code `iLIDS.py`
  - The data structure should look like

  ```
  QMUL-iLIDS/
  ├── images/
  ├── splits.json (This will be created by `iLIDS.py` in `fastreid/data/datasets/` folder)
  ```

- (9) VIPer
  - Create a directory called `viper`
  - Download `viper` from [link](https://users.soe.ucsc.edu/~manduchi/VIPeR.v1.0.zip) and extract the files.
  - Split sets can be created by my matlab code `make_split.m` (this code is included in the datasets folder)
  - The data structure should look like
  ```
  viper/
  ├── cam_a/
  ├── cam_b/
  ├── make_split.m (my matlab code)
  ├── split_1a # Train: split1, Test: split2 ([query]cam1->[gallery]cam2)
  ├── split_1b # Train: split2, Test: split1 (cam1->cam2)
  ├── split_1c # Train: split1, Test: split2 (cam2->cam1)
  ├── split_1d # Train: split2, Test: split1 (cam2->cam1)
  ...
  ...
  ├── split_10a
  ├── split_10b
  ├── split_10c
  ├── split_10d
  ```

## 2) Train

`python3 ./tools/train_net.py --config-file ./configs/Sample/DG-mobilenet.yml`


- Evaluation only

`python3 ./tools/train_net.py --config-file ./configs/Sample/DG-mobilenet.yml --eval-only`


