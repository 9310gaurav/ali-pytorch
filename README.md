# Adversarially Learned Inference
PyTorch implementation of [Aversarially Learned Inference](https://arxiv.org/abs/1606.00704)

## SVHN samples
### After 1 epoch : 
<img src="https://github.com/9310gaurav/ali-pytorch/blob/master/saved_images_svhn/fake_0.png" width="480">

### After 10 epochs :
<img src="https://github.com/9310gaurav/ali-pytorch/blob/master/saved_images_svhn/fake_10.png" width="480">

### After 40 epochs :
<img src="https://github.com/9310gaurav/ali-pytorch/blob/master/saved_images_svhn/fake_40.png" width="480">

## CIFAR10 samples :
### After 1 epoch :
<img src="https://github.com/9310gaurav/ali-pytorch/blob/master/saved_images_cifar/fake_0.png" width="480">

### After 100 epochs :
<img src="https://github.com/9310gaurav/ali-pytorch/blob/master/saved_images_cifar/fake_100.png" width="480">

### After 500 epochs :
<img src="https://github.com/9310gaurav/ali-pytorch/blob/master/saved_images_cifar/fake.png" width="480">

## To test SVHN pretrained embeddings for semi-supervised learning using L2-SVMs :
```python3.5 test_semisup.py --dataset=svhn --dataroot=<dataroot> --model_path=<saved_model_path>```

Note : The provided model was trained for 100 epochs and gives an error rate of 23% as opposed to 19.5% reported in the paper. The original model's training was not stable in this implementation and I've used some GAN hacks like adding instance noise and selective training. Refer to this for more details : https://github.com/soumith/ganhacks.

## Cite
```
@article{DBLP:journals/corr/DumoulinBPLAMC16,
  author    = {Vincent Dumoulin and
               Ishmael Belghazi and
               Ben Poole and
               Alex Lamb and
               Mart{\'{\i}}n Arjovsky and
               Olivier Mastropietro and
               Aaron C. Courville},
  title     = {Adversarially Learned Inference},
  journal   = {CoRR},
  volume    = {abs/1606.00704},
  year      = {2016},
  url       = {http://arxiv.org/abs/1606.00704},
}
```
