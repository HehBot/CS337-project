# CS337 project
## Proposal
Neural networks have historically been black-boxes; their inner working inscrutable to analysis. In this paper techniques to visualise the action of a neural network are explored. Using gradient ascent with regularization, input images that maximise the activation of a particular node in a convnet can be generated, and these provide insight into the way the neural network processes images by understanding excatly what each neuron has learned and thus what computation it performs. Adding to this we hope to explore the adversarial aspect of input engineering for convnets, creating images that differ imperceptibly from "good" images and yet are predicted incorrectly by the convnet with high confidence.

## Datasets
ImageNet

## Paper
Yosinski et al, _Understanding Neural Networks Through Deep Visualization_ (DOI 1506.06579)

## TODO (in order of priority from highest to least)
1. The authors say, "Our network was trained on ImageNet by first subtracting the per-pixel mean of examples in ImageNet before inputting training examples to the network. Thus, the direct input to the network, x, can be thought of as a zero-centered input." However, our current code generates pixel values in $[0, 1)$, and the pre-trained model was trained on images with non-zero mean. Modify the code to train AlexNet on zero-mean images, and then generate random image from a standard gaussian instead of $[0, 1)$.
2. [DONE] Implement the rest of the regularizers from the paper.
3. Try installing the toolbox, if you can (https://github.com/yosinski/deep-visualization-toolbox)
