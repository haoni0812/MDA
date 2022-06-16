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




## 2) Train

`python3 ./tools/train_net.py --config-file ./configs/Sample/DG-mobilenet.yml`


- Evaluation only

`python3 ./tools/train_net.py --config-file ./configs/Sample/DG-mobilenet.yml --eval-only`


