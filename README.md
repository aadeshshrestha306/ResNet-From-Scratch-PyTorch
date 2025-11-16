# Implementing ResNet Paper from Scratch in PyTorch


Based on the paper [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385)


The 34 and 50 layer version, is trained on CIFAR-10 dataset for 32 epochs(12500 steps) with a batch size of 128. After 3000 steps or approximately 8 epochs, the learning rate is reduced by 10 times. Here is a comparison between ResNet-34 and ResNet-50.

<table>
    <tr>
        <th></th>
        <th>Epochs</th>
        <th>Loss</th>
        <th>Accuracy</th>
    </tr>
    <tbody>
        <tr>
            <td>ResNet-34</td>
            <td>8</td>
            <td>0.4884</td>
            <td>80.99%</td>
        </tr>
        <tr>
            <td>ResNet-50</td>
            <td>8</td>
            <td>0.4524</td>
            <td>78.03%</td>
        </tr>
        <tr>
            <td colspan="4"></td>
        </tr>
        <tr>
            <td>ResNet-34</td>
            <td>16</td>
            <td>0.5332</td>
            <td>88.10%</td>
        </tr>
        <tr>
            <td>ResNet-50</td>
            <td>16</td>
            <td>0.3884</td>
            <td>86.95%</td>
        </tr>
        <tr>
            <td colspan="4"></td>
        </tr>
        <tr>
            <td>ResNet-34</td>
            <td>24</td>
            <td>0.3453</td>
            <td>89.13%</td>
        </tr>
        <tr>
            <td>ResNet-50</td>
            <td>24</td>
            <td>0.3535</td>
            <td>86.95%</td>
        </tr>
        <tr>
            <td colspan="4"></td>
        </tr>
        <tr>
            <td>ResNet-34</td>
            <td>32</td>
            <td>0.2230</td>
            <td>89.26%</td>
        </tr>
        <tr>
            <td>ResNet-50</td>
            <td>32</td>
            <td>0.3747</td>
            <td>87.07%</td>
        </tr>
        <tr>
            <td colspan="4"></td>
        </tr>
    </tbody>
</table>
