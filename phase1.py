import torch
import torch.optim as optim
from torchvision import models
from PIL import Image
import numpy as np

class_idx=456       # bow
reg = 'none'
reg_strength = 0.01

# Set the random seed for CPU operations
torch.manual_seed(42)

# Load the pre-trained AlexNet model
alexnet = models.alexnet(pretrained=True)
alexnet.eval()

# Create a random input image (you can also start with an existing image)
input_image = torch.rand((1, 3, 224, 224),  requires_grad=True)

learning_rate = 1
num_iterations = 1000

optimizer = optim.SGD([input_image], lr=learning_rate)

for iteration in range(num_iterations):
    # Forward pass to compute activation
    output = alexnet(input_image)
    activation = output[0, class_idx]  # Replace 'class_idx' with the index of your target class

    # Backward pass to compute gradient
    alexnet.zero_grad()
    activation.backward()

    # Update the input image with a fraction of the gradient
    if (reg == 'l2'):
        input_image.data += learning_rate * input_image.grad.data - 2 * reg_strength * input_image.data
    elif (reg == 'l1'):
        input_image.data += learning_rate * input_image.grad.data - reg_strength * np.sign(input_image.data)
    elif (reg == 'none'):
        input_image.data += learning_rate * input_image.grad.data
    

    # Clamp pixel values to be within [0, 1] or [0, 255] depending on your input range
    input_image.data = torch.clamp(input_image.data, 0, 1)  # Assuming input image is in [0, 1] range

    # Zero out the gradient for the next iteration
    input_image.grad.zero_()

    print(f"Iteration {iteration + 1}/{num_iterations}, Activation: {activation.item()}")

# Forward pass on the optimized image to get class probabilities
with torch.no_grad():
    output = alexnet(input_image)

# Get the predicted class label
predicted_class = output.argmax().item()

# Get the confidence (probability) of the predicted class
confidence = torch.softmax(output, dim=1)[0, predicted_class].item()

print(f"Predicted Class: {predicted_class}, Confidence: {confidence}")

# Convert the optimized tensor back to a PIL image if needed
optimized_image = (input_image * 255).byte()
optimized_image = optimized_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
optimized_image = Image.fromarray(optimized_image)
optimized_image.show()