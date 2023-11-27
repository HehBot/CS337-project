import torch
import torch.optim as optim
from torchvision import models
from torchvision.transforms import GaussianBlur
from PIL import Image
import numpy as np
import os

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

class TargetedAttack:
    def __init__(
        self,
        class_idx,
        reg="none",
        theta_decay=0.01,
        theta_b_width=3,
        theta_n_pct=5,
        theta_c_pct=95,
        learning_rate=1,
        num_iterations=1000,
        lmbda = 0.01,
        alfa = 0,
    ):
        self.class_idx = class_idx
        self.reg = reg
        self.theta_decay = theta_decay
        self.theta_b_width = theta_b_width
        self.theta_n_pct = theta_n_pct
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta_c_pct = theta_c_pct
        self.lmbda = lmbda
        self.alfa = alfa

        # Set the random seed for CPU operations
        torch.manual_seed(42)

        # Load the pre-trained AlexNet model
        self.alexnet = models.alexnet(pretrained=True)
        self.alexnet.eval()

        # Create a random input image (you can also start with an existing image)
        self.input_image = torch.randn((1, 3, 224, 224)) + mean[None, :, None, None]
        self.input_image.requires_grad = True

        # Create an optimizer
        self.optimizer = optim.SGD([self.input_image], lr=self.learning_rate)

    def forward_pass(self):
        output = self.alexnet(self.input_image)
        # print("size of output:",output.shape)
        activation = output[0, self.class_idx]
        return activation

    def backward_pass(self):
        self.alexnet.zero_grad()
        activation = self.forward_pass()
        activation.backward()
    def compute_pixel_contributions(self):
        gradients = self.input_image.grad.data.clone()
        # self.alexnet.zero_grad()  # Clear the gradients for subsequent passes

        return gradients

    def clip_pixels_with_small_contributions(self, contributions, percentile):
        threshold = np.percentile(contributions.abs().numpy(), percentile)
        mask = contributions.abs() <= threshold
        self.input_image.data = torch.where(mask, self.input_image.data, torch.tensor(0))
    

    def update_input_image(self):
        self.input_image.data += self.learning_rate * self.input_image.grad.data
        if self.reg == "l2":
            self.input_image.data += -2 * self.theta_decay * (self.input_image.data - mean[None, :, None, None])
            # self.input_image.data = torch.clamp(self.input_image.data, 0, 1)
        elif self.reg == "l1":
            self.input_image.data += -self.theta_decay * np.sign(self.input_image.data - mean[None, :, None, None])
            # self.input_image.data = torch.clamp(self.input_image.data, 0, 1)
        elif self.reg == "gaussian_blur":
            self.input_image.data = GaussianBlur(self.theta_b_width)(self.input_image.data)
            # self.input_image.data = torch.clamp(self.input_image.data, 0, 1)
        elif self.reg == "none":
            # self.input_image.data = torch.clamp(self.input_image.data, 0, 1)
            pass
        elif self.reg == "mix":
            self.input_image.data += -self.lmbda * self.alfa * np.sign(self.input_image.data - mean[None, :, None, None])
            self.input_image.data += -2 * (self.lmbda * (1 - self.alfa)) * (self.input_image.data - mean[None, :, None, None])
            self.input_image.data = GaussianBlur(self.theta_b_width)(self.input_image.data)
        else:
            print("INCORRECT REGULARIZATION\n")
            exit()

    def optimize(self):
        for iteration in range(self.num_iterations):
            self.backward_pass()

            self.update_input_image()

            # Clip pixels with small contributions
            contributions = self.compute_pixel_contributions()
            self.clip_pixels_with_small_contributions(contributions, self.theta_c_pct)  # Adjust percentile as needed

            # Clip pixels with small norm
            norm_threshold = np.percentile(self.input_image.data.numpy(), self.theta_n_pct)
            self.input_image.data = torch.where(
                torch.abs(self.input_image.data) > norm_threshold,
                self.input_image.data,
                torch.tensor(0),
            )
            self.input_image.grad.zero_()
            print(
                f"Iteration {iteration + 1}/{self.num_iterations}, Activation: {self.forward_pass().item()}"
            )
    def optimize_with_grid_search(self):
        activation = []
        for iteration in range(self.num_iterations):
            self.backward_pass()

            self.update_input_image()

            # Clip pixels with small contributions
            contributions = self.compute_pixel_contributions()
            self.clip_pixels_with_small_contributions(contributions, self.theta_c_pct)  # Adjust percentile as needed

            # Clip pixels with small norm
            norm_threshold = np.percentile(self.input_image.data.numpy(), self.theta_n_pct)
            self.input_image.data = torch.where(
                torch.abs(self.input_image.data) > norm_threshold,
                self.input_image.data,
                torch.tensor(0),
            )
            self.input_image.grad.zero_()
            activation.append(self.forward_pass().item())
        return activation

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
        os.makedirs("figures", exist_ok=True)
        save_path = os.path.join("figures", f"alpha_{self.alfa}clip_cutoff{self.theta_n_pct}_blur_size{self.theta_b_width}percentile{self.theta_c_pct}.png")
        optimized_image.save(save_path)
        return optimized_image

def grid_search(theta_c_pct,class_idx,reg,theta_b_width, theta_n_pct,learning_rate,num_iterations,alfa,lmbda):
    activation = []
    predicted_class = 0
    confidence = 0
    optimized_image = None
    p=0 
    b=0
    c =0 
    a =0
    for percentile in theta_c_pct:
      for blur_size in theta_b_width:
        for clip_cutoff in theta_n_pct:
          for alp in alfa:
              theta_decay = 0
              attack = TargetedAttack(
                  class_idx,
                  reg,
                  theta_decay,
                  blur_size,
                  clip_cutoff,
                  percentile,
                  learning_rate,
                  num_iterations,
                  lmbda,
                  alp,
              )
              attack.optimize_with_grid_search()
              optimized_image = attack.get_optimized_image()
              predicted, conf = attack.get_prediction()
              optimized_image.show()
    for i in range(len(activation)):
      print(
            f"Iteration {i + 1}/{len(activation)}, Activation: {activation[i]}"
      )
      
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence}")
    print(f"Percentile: {p}, alpha:{a}, Blur_size:{b}, clip_cutoff:{c}")
  


if __name__ == "__main__":
    class_idx = 736  # Replace with the index of your target class
    reg = "mix"  # Change the regularization method if needed
    theta_decay = 0.01  # Adjust the regularization strength as needed
    theta_b_width = [1, 3, 6]
    theta_n_pct = [0, 0.1, 0.2]
    learning_rate = 1
    num_iterations = 200
    theta_c_pct = [100, 99.9, 99]
    lmbda = 0.01
    alfa = [0, 0.2, 0.4]
    
    grid_search(theta_c_pct,class_idx,reg,theta_b_width, theta_n_pct,learning_rate,num_iterations,alfa,lmbda)
    
    
# if __name__ == "__main__":
#     class_idx = 736  # Replace with the index of your target class
#     reg = "mix"  # Change the regularization method if needed
#     theta_decay = 0.01  # Adjust the regularization strength as needed
#     theta_b_width = [3,5]
#     theta_n_pct = [0,0.1,0.2]
#     learning_rate = 1
#     num_iterations = 200
#     theta_c_pct = [100,99.9,99]
#     lmbda = 0.01
#     alfa = [0,.2,.4]
    
#     grid_search(theta_c_pct,class_idx,reg,theta_b_width, theta_n_pct,learning_rate,num_iterations,alfa,lmbda)