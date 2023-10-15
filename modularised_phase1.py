import torch
import torch.optim as optim
from torchvision import models
from PIL import Image
import numpy as np

class TargetedAttack:
    def __init__(self, class_idx, reg='none', reg_strength=0.01, learning_rate=1, num_iterations=1000):
        self.class_idx = class_idx
        self.reg = reg
        self.reg_strength = reg_strength
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

        # Set the random seed for CPU operations
        torch.manual_seed(42)

        # Load the pre-trained AlexNet model
        self.alexnet = models.alexnet(pretrained=True)
        self.alexnet.eval()

        # Create a random input image (you can also start with an existing image)
        self.input_image = torch.rand((1, 3, 224, 224), requires_grad=True)

        # Create an optimizer
        self.optimizer = optim.SGD([self.input_image], lr=self.learning_rate)

    def forward_pass(self):
        output = self.alexnet(self.input_image)
        activation = output[0, self.class_idx]
        return activation

    def backward_pass(self):
        self.alexnet.zero_grad()
        activation = self.forward_pass()
        activation.backward()

    def update_input_image(self):
        if self.reg == 'l2':
            self.input_image.data += self.learning_rate * self.input_image.grad.data - 2 * self.reg_strength * self.input_image.data
        elif self.reg == 'l1':
            self.input_image.data += self.learning_rate * self.input_image.grad.data - self.reg_strength * np.sign(self.input_image.data)
        elif self.reg == 'none':
            self.input_image.data += self.learning_rate * self.input_image.grad.data
        self.input_image.data = torch.clamp(self.input_image.data, 0, 1)

    def optimize(self):
        for iteration in range(self.num_iterations):
            self.backward_pass()
            self.update_input_image()
            self.input_image.grad.zero_()
            print(f"Iteration {iteration + 1}/{self.num_iterations}, Activation: {self.forward_pass().item()}")

    def get_prediction(self):
        with torch.no_grad():
            output = self.alexnet(self.input_image)

        predicted_class = output.argmax().item()
        confidence = torch.softmax(output, dim=1)[0, predicted_class].item()
        return predicted_class, confidence

    def get_optimized_image(self):
        optimized_image = (self.input_image * 255).byte()
        optimized_image = optimized_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        optimized_image = Image.fromarray(optimized_image)
        return optimized_image

if __name__ == "__main__":
    class_idx = 456  # Replace with the index of your target class
    reg = 'l2'  # Change the regularization method if needed
    reg_strength = 0.01  # Adjust the regularization strength as needed
    learning_rate = 1
    num_iterations = 1000

    attack = TargetedAttack(class_idx, reg, reg_strength, learning_rate, num_iterations)
    attack.optimize()

    predicted_class, confidence = attack.get_prediction()
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence}")

    optimized_image = attack.get_optimized_image()
    optimized_image.show()
