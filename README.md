# CS337 project
## Proposal
Neural networks have historically been black-boxes; their inner working inscrutable to analysis. In this paper techniques to visualise the action of a neural network are explored. Using gradient ascent with regularization, input images that maximise the activation of a particular node in a convnet can be generated, and these provide insight into the way the neural network processes images by understanding excatly what each neuron has learned and thus what computation it performs. Adding to this we hope to explore the adversarial aspect of input engineering for convnets, creating images that differ imperceptibly from "good" images and yet are predicted incorrectly by the convnet with high confidence.

## Datasets
ImageNet

## Paper
Yosinski et al, _Understanding Neural Networks Through Deep Visualization_ (DOI 1506.06579)

## Presentation and Report
INSERT LINK

INSERT LINK

## Description of files
The visualise_output_neuron.py script is designed for visualizing and optimizing the input image to maximise the activation of a specific neuron in a pre-trained AlexNet model. The goal is to generate an image that strongly activates the specified neuron, providing insight into what features the neuron is responsive to.

The visualise_hidden_neuron.py script is designed for visualizing and optimizing an input image to maximize the activation of a specific neuron in a hidden layer of a pre-trained AlexNet model. 

## Running instructions
### 1. visualise_output_neuron.py
```
python3 visualise_output_neuron.py <desired_class_index> <desired_regularization>
```
Replace <desired_class_index> with the index of the class you want to visualise, and <desired_regularization> with the desired regularization method (l1, l2, gaussian_blur, clip, mix, or none).
### 2. visualise_hidden_neuron.py
```
python visualise_hidden_neuron.py <desired_layer> <desired_node> <desired_regularization>
```
where $0 <=$ desired_layer $<= 12$

Replace <desired_layer> and <desired_node> with the layer number and node number of the target neuron you want to visualise, and <desired_regularization> with the desired regularization method (l1, l2, gaussian_blur, clip, mix, or none).

Adjust the hyperparameters in the script (in main) if needed. The relevant hyperparameters for each regularization technique is written in the comments for your convenience

### 3. grid_search.py
```
python grid_search.py
cd figures
python tile.py
```
The grid_search.py script is designed to perform a grid search over a set of hyperparameters for the visualization of an output layer neuron.  Adjust the hyperparameters in the script as needed.

All the visualizations are stored in the "figures" folder. Running tile.py tiles all the visualizations into a single grid image (will be saved at "figures/tiled_image.png").

### 4. attack.py
ARNAV