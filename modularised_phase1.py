import torch
import torch.optim as optim
from torchvision import models
from PIL import Image
import numpy as np

class_idx = 456  # Class index for "bow"
reg = 'none'
reg_strength = 0.01

def load_pretrained_model():
    # Load the pre-trained AlexNet model
    alexnet = models.alexnet(pretrained=True)
    alexnet.eval()
    return alexnet

def create_input_image():
    # Create a random input image (you can also start with an existing image)
    input_image = torch.rand((1, 3, 224, 224), requires_grad=True)
    return input_image

def update_input_image(input_image, learning_rate, reg, reg_strength):
    # Update the input image with gradient and optional regularization
    input_image.data += learning_rate * input_image.grad.data
    if reg == 'l2':
        input_image.data -= 2 * reg_strength * input_image.data
    elif reg == 'l1':
        input_image.data -= reg_strength * np.sign(input_image.data)
    return torch.clamp(input_image, 0, 1)  # Clamp pixel values to [0, 1] range

def optimize_image(input_image, model, class_idx, learning_rate, num_iterations, reg, reg_strength):
    optimizer = optim.SGD([input_image], lr=learning_rate)

    for iteration in range(num_iterations):
        # Forward pass to compute activation
        output = model(input_image)
        activation = output[0, class_idx]

        # Backward pass to compute gradient
        model.zero_grad()
        activation.backward()

        # Update the input image
        input_image = update_input_image(input_image, learning_rate, reg, reg_strength)

        # Zero out the gradient for the next iteration
        input_image.grad.zero_()

        print(f"Iteration {iteration + 1}/{num_iterations}, Activation: {activation.item()}")

    return input_image

def main():
    # Set the random seed for CPU operations
    torch.manual_seed(42)

    # Load the pre-trained model
    alexnet = load_pretrained_model()

    # Create the input image
    input_image = create_input_image()

    learning_rate = 1
    num_iterations = 1000

    # Optimize the input image
    optimized_image = optimize_image(input_image, alexnet, class_idx, learning_rate, num_iterations, reg, reg_strength)

    # Forward pass on the optimized image to get class probabilities
    with torch.no_grad():
        output = alexnet(optimized_image)

    # Get the predicted class label
    predicted_class = output.argmax().item()

    # Get the confidence (probability) of the predicted class
    confidence = torch.softmax(output, dim=1)[0, predicted_class].item()

    print(f"Predicted Class: {predicted_class}, Confidence: {confidence}")

    # Convert the optimized tensor back to a PIL image
    optimized_image = (optimized_image * 255).byte()
    optimized_image = optimized_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    optimized_image = Image.fromarray(optimized_image)
    optimized_image.show()

if __name__ == "__main__":
    main()
