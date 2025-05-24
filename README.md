# CFReID: Continual Few-shot Person Re-Identification

This is the official implementation of our paper:

**CFReID: Continual Few-shot Person Re-Identification**  
Hao Ni, Jingkuan Song, et al.  
[[arXiv](https://arxiv.org/abs/2503.18469)]

This work is an extension of our CVPR 2022 paper:  
**Meta Distribution Alignment for Generalizable Person Re-Identification (MDA)**  
[[IEEE](https://ieeexplore.ieee.org/document/9880010)]

---

## 🛠️ 1. Prerequisites

- Ubuntu 18.04+
- Python 3.8+
- PyTorch ≥ 1.8
- NVIDIA GPU (with ≥ 11GB memory)
- Anaconda (recommended)
- CUDA 10.2/11.x with compatible cuDNN


## 📁 2. Dataset Preparation

Create a `datasets/` directory in the root and prepare datasets as follows:

### Market-1501

```
data/
└── Market-1501-v15.09.15/
    ├── bounding_box_train/
    ├── bounding_box_test/
    └── query/
```

### DukeMTMC-reID

```
data/
└── DukeMTMC-reID/
    ├── bounding_box_train/
    ├── bounding_box_test/
    └── query/
```

> For other datasets (e.g., MSMT17, SYSU-MM01), follow the same structure as in the [MDA repo](https://github.com/Nihaoooo/MDA-ReID) or refer to our dataset instructions.

---

## 🚀 3. Training

To train the model under a continual few-shot setting, run:

```bash
bash run_train.sh
```

You can modify `run_train.sh` to configure:

* Datasets
* Continual stages
* Few-shot sample sizes
* Training hyperparameters

---

## 📊 4. Evaluation

To evaluate a trained model, run:

```bash
bash run_evaluate.sh
```

This script loads the latest checkpoint and performs evaluation on the specified dataset.

---


## 📚 5. Citation

If you find our work useful, please cite:

```bibtex
@article{ni2024cfreid,
  title={CFReID: Continual Few-shot Person Re-Identification},
  author={Ni, Hao and Song, Jingkuan and others},
  journal={arXiv preprint arXiv:2503.18469},
  year={2024}
}

@inproceedings{ni2022meta,
  title={Meta Distribution Alignment for Generalizable Person Re-Identification},
  author={Ni, Hao and Song, Jingkuan and others},
  booktitle={CVPR},
  year={2022}
}
```

---

## 🙏 6. Acknowledgments

This codebase builds on [FastReID](https://github.com/JDAI-CV/fast-reid) and extends our previous work on meta distribution alignment. We thank the open-source community for their valuable tools and resources.

---

