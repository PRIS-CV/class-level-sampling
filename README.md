# class-level-sampling

Code release for the paper "Understanding Episode Hardness in Few-shot Learning" (Submited to IEEE TPAMI).

## :heavy_check_mark: Requirements
* Ubuntu 16.04
* Python 3.7
* [CUDA 11.0](https://developer.nvidia.com/cuda-toolkit)
* [PyTorch 1.7.1](https://pytorch.org)


## :Backbone Training for the ProtoNet
We use the same backbone network and training strategies as 'RENet'. Please refer to https://github.com/dahyun-kang/renet for the backbone training.




## :fire: Training scripts
To train in the 5-way K-shot setting:
```bash
bash scripts/train/{dataset_name}_5wKs.sh
```
For example, to train ReNet on the CUB dataset in the 5-way 1-shot setting:
```bash
bash scripts/train/cub_5w1s.sh
```

To test in the 5-way K-shot setting:
```bash
bash scripts/test/{dataset_name}_5wKs.sh
```
For example, to test ReNet on the miniImagenet dataset in the 5-way 1-shot setting:
```bash
bash scripts/test/miniimagenet_5w1s.sh
```
