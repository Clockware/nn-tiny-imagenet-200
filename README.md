# NN for tiny-imagenet-200

- tiny-imagenet-200 (64x64xRGB)
- No pretrained weights
- 43% accuracy (ratio)

#### Env notes:
- conda create -n NAME python=3.6
- conda install scikit-learn
- conda install pytorch=0.4.1 cuda90 -c pytorch
- conda install torchvision

#### Expected typical output:

Device: cuda

CUDA Version: 9.0.176

Downloading tiny-imagenet-200... ./tiny-imagenet-200.zip

Done.

Weight shapes: [torch.Size([64, 3, 3, 3]), torch.Size([64]), torch.Size([64]), torch.Size([128, 64, 3, 3]), torch.Size([128]), torch.Size([128]), torch.Size([128, 128, 3, 3]), torch.Size([128]), torch.Size([128]), torch.Size([256, 128, 3, 3]), torch.Size([256]), torch.Size([256]), torch.Size([256, 256, 3, 3]), torch.Size([256]), torch.Size([256]), torch.Size([512, 256, 3, 3]), torch.Size([512]), torch.Size([512]), torch.Size([200, 8192]), torch.Size([200])]

Calculating weights... Done.

================================================== Epoch #0 | 14.97% accuracy
================================================== Epoch #1 | 20.82% accuracy
================================================== Epoch #2 | 25.86% accuracy
================================================== Epoch #3 | 30.29% accuracy
================================================== Epoch #4 | 31.89% accuracy
================================================== Epoch #5 | 33.99% accuracy
================================================== Epoch #6 | 35.83% accuracy
================================================== Epoch #7 | 37.82% accuracy
================================================== Epoch #8 | 37.72% accuracy
================================================== Epoch #9 | 40.07% accuracy
================================================== Epoch #10 | 40.96% accuracy
================================================== Epoch #11 | 42.01% accuracy
================================================== Epoch #12 | 42.41% accuracy
================================================== Epoch #13 | 42.55% accuracy
================================================== Epoch #14 | 43.19% accuracy
================================================== Epoch #15 | 43.00% accuracy
================================================== Epoch #16 | 44.09% accuracy
================================================== Epoch #17 | 44.37% accuracy
================================================== Epoch #18 | 45.24% accuracy
================================================== Epoch #19 | 44.41% accuracy
================================================== Epoch #20 | 44.41% accuracy

Time is against us!

Train: 71.58% accuracy

Val: 44.13% accuracy

Test: 44.74% accuracy
