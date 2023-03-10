# class-level-sampling

Code release for the paper "Understanding Episode Hardness in Few-shot Learning" (Submited to IEEE TPAMI).

In this paper, we propose a new metric named Inverse-Fisher Discriminant Ratio (IFDR) to assess episode hardness. And we propose a class-level hard episode sampling schemes (cl-sampling) that can be augmented to any existing few-shot learning approaches to boost their performance. Delving deeper, we also develop a novel variant of cl-sampling -- class-pair-level hard episode sampling scheme (cpl-sampling), which not only guarantees the generation of hard episode distribution like the class-level approach but also significantly reduces the time-cost.

## * Requirements
* Ubuntu 16.04
* Python 3.7
* [CUDA 11.0](https://developer.nvidia.com/cuda-toolkit)
* [PyTorch 1.7.1](https://pytorch.org)


## *  Backbone Training for the ProtoNet
We use the same backbone network and training strategies as 'RENet'. Please refer to https://github.com/dahyun-kang/renet for the backbone training.


## *  Training Pipeline
The meta-learning model trained consists of two main training phases in order to achieve effective few-shot classifiers. The meta-learning model firstly is trained using randomly sampling episodes. Then we intentionally pick up hard episodes using cl-sampling or cpl-sampling for fine-tuning the model.


## *  Training scripts
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

### Random-sampling training
Edit scripts/train/{dataset_name}_5wKs.sh in the repository to change '-train_sampling' to random-sampling'.
And then you can train the model using randomly sampling episodes by Training scripts.


### Fine-tuning 
Load the pretrained model. 
Edit scripts/train/{dataset_name}_5wKs.sh in the repository to change '-train_sampling' to cl-sampling/cpl-sampling'.
And then you can train the model using cl-sampling or cpl-sampling by Training scripts.

## *  Acknowledgement
We adopted the main code bases from [RENet]([https://github.com/dahyun-kang/renet]), and we really appreciate it.

## * Contact
Thanks for your attention! If you have any suggestion or question, you can leave a message here or contact us directly:

* guoyurong@bupt.edu.cn
* mazhanyu@bupt.edu.cn
