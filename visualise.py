from sys import argv, exit
import torch
import torch.optim as optim
from torchvision.transforms import GaussianBlur
from torchvision.utils import save_image
import numpy as np
from tqdm import trange

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])


class Visualise:
    def __init__(
        self,
        model,
        required_class,
        reg="none",
        theta_decay=0.01,
        theta_b_width=3,
        theta_n_pct=5,
        theta_c_pct=5,
        learning_rate=1,
        num_iterations=1000,
        lmbda=0.01,
        alfa=0,
    ):
        self.required_class = required_class
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
        self.model = model
        self.model.eval()

        # Create a random input image (you can also start with an existing image)
        self.input_image = torch.randn((1, 3, 224, 224)) + mean[None, :, None, None]
        self.input_image.requires_grad = True

        # Create an optimizer
        self.optimizer = optim.SGD([self.input_image], lr=self.learning_rate)

    def forward_pass(self):
        output = self.model(self.input_image)
        # print("size of output:",output.shape)
        activation = output[0, self.required_class]
        return activation

    def backward_pass(self):
        self.model.zero_grad()
        activation = self.forward_pass()
        activation.backward()

    def compute_pixel_contributions(self):
        gradients = self.input_image.grad.data.clone()
        # self.model.zero_grad()  # Clear the gradients for subsequent passes

        return gradients

    def clip_pixels_with_small_contributions(self, contributions, percentile):
        threshold = np.percentile(contributions.abs().numpy(), percentile)
        mask = contributions.abs() >= threshold
        self.input_image.data = torch.where(
            mask, self.input_image.data, torch.tensor(0)
        )

    def update_input_image(self):
        self.input_image.data += self.learning_rate * self.input_image.grad.data
        if self.reg == "l2":
            self.input_image.data += (
                -2
                * self.theta_decay
                * (self.input_image.data - mean[None, :, None, None])
            )
            self.input_image.data = torch.clamp(self.input_image.data, 0, 1)
        elif self.reg == "l1":
            self.input_image.data += -self.theta_decay * np.sign(
                self.input_image.data - mean[None, :, None, None]
            )
            self.input_image.data = torch.clamp(self.input_image.data, 0, 1)
        elif self.reg == "gaussian_blur":
            self.input_image.data = GaussianBlur(self.theta_b_width)(
                self.input_image.data
            )
            self.input_image.data = torch.clamp(self.input_image.data, 0, 1)
        elif self.reg == "none":
            self.input_image.data = torch.clamp(self.input_image.data, 0, 1)
        elif self.reg == "mix":
            self.input_image.data += (
                -self.lmbda
                * self.alfa
                * np.sign(self.input_image.data - mean[None, :, None, None])
            )
            self.input_image.data += (
                -2
                * (self.lmbda * (1 - self.alfa))
                * (self.input_image.data - mean[None, :, None, None])
            )
            self.input_image.data = GaussianBlur(self.theta_b_width)(
                self.input_image.data
            )
            self.input_image.data = torch.clamp(self.input_image.data, 0, 1)
        else:
            print("INCORRECT REGULARIZATION\n")
            exit(1)

    def optimize(self):
        t = trange(self.num_iterations)
        for iteration in t:
            self.backward_pass()

            self.update_input_image()

            # Clip pixels with small contributions
            contributions = self.compute_pixel_contributions()
            self.clip_pixels_with_small_contributions(
                contributions, self.theta_c_pct
            )  # Adjust percentile as needed

            # Clip pixels with small norm
            norm_threshold = np.percentile(
                self.input_image.data.numpy(), self.theta_n_pct
            )
            self.input_image.data = torch.where(
                torch.abs(self.input_image.data) > norm_threshold,
                self.input_image.data,
                torch.tensor(0),
            )
            self.input_image.grad.zero_()
            t.set_description(f"Activation: {self.forward_pass().item():.2f}")
            t.refresh()

    def get_prediction(self):
        with torch.no_grad():
            output = self.model(self.input_image)

        predicted_class = output.argmax().item()
        confidence = torch.softmax(output, dim=1)[0, predicted_class].item()
        return predicted_class, confidence

    def get_optimized_image(self):
        return self.input_image.squeeze(0)


if __name__ == "__main__":
    if len(argv) != 2:
        print("Usage: python %s <desired_class>")
        exit(1)

    reg = "mix"  # Change the regularization method if needed
    theta_decay = 0.01  # Adjust the regularization strength as needed
    theta_b_width = 3
    theta_n_pct = 0.1
    learning_rate = 1
    num_iterations = 200
    theta_c_pct = 5
    lmbda = 0.01
    alfa = 0

    from torchvision.models import alexnet, AlexNet_Weights

    weights = AlexNet_Weights.DEFAULT
    model = alexnet(weights=weights)
    class_names = weights.meta["categories"]

    required_class = int(argv[1])

    attack = Visualise(
        model,
        required_class,
        reg,
        theta_decay,
        theta_b_width,
        theta_n_pct,
        theta_c_pct,
        learning_rate,
        num_iterations,
        lmbda,
        alfa,
    )
    attack.optimize()

    predicted_class, confidence = attack.get_prediction()
    print(f"Desired Class:   {class_names[required_class]}")
    print(f"Predicted Class: {class_names[predicted_class]}, Confidence: {confidence}")

    optimized_image = attack.get_optimized_image()
    save_image(
        optimized_image, f"{class_names[predicted_class]}_visualise.png", format="png"
    )
