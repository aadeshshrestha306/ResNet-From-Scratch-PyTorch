# Implementing ResNet Paper from Scratch in PyTorch


Based on the paper [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385)

The 34 layer version of ResNet, is trained on CIFAR-10 dataset with four learning rate values [0.1, 0.01, 0.001, 0.0001]. After 5 epochs, the loss and accuracy were as follows:

lr = 0.1 -> loss = 1.08363, accuracy = 73.29%<br>
lr = 0.01 -> loss = 0.9703, accuracy = 76.01%<br>
lr = 0.001 -> loss = 0.7687, accuracy = 66.28%<br>
lr = 0.0001 -> loss = 1.9381, accuracy = 40.16%<br>

Hyperparameters for the network is assigned as per the paper.