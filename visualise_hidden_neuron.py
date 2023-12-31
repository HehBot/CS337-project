from sys import argv, exit
import torch
from torchvision.transforms import GaussianBlur
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.utils import save_image
from tqdm import trange

mean = torch.tensor([0.485, 0.456, 0.406])


class Visualise:
    def __init__(
        self,
        model,
        hidden_node,
        reg="none",
        theta_decay=0.01,
        theta_b_width=3,
        theta_n_pct=5,
        theta_c_pct=5,
        learning_rate=1,
        num_iterations=1000,
        alfa=0,
        hidden_layer=4,
    ):
        self.hidden_node = hidden_node
        self.reg = reg
        self.theta_decay = theta_decay
        self.theta_b_width = theta_b_width
        self.theta_b_every = theta_b_every
        self.theta_n_pct = theta_n_pct
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta_c_pct = theta_c_pct
        self.alfa = alfa

        self.gaussian_count = 0

        # Load the pre-trained AlexNet model
        self.model = model
        self.model.eval()

        # Create a random input image (you can also start with an existing image)
        self.input_image = torch.randn((1, 3, 224, 224)) + mean[None, :, None, None]
        self.input_image.requires_grad = True

        return_nodes = {f"features.{hidden_layer}": f"features.{hidden_layer}"}
        self.sub_model = create_feature_extractor(self.model, return_nodes=return_nodes)

    def forward_pass(self):
        intermediate_outputs = self.sub_model(self.input_image)
        # print(intermediate_outputs)

        activation = torch.norm(
            intermediate_outputs[f"features.{hidden_layer}"][0, self.hidden_node]
        )
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
        threshold = torch.quantile(contributions.abs(), percentile / 100)
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
            self.input_image.data += -self.theta_decay * torch.sign(
                self.input_image.data - mean[None, :, None, None]
            )
            self.input_image.data = torch.clamp(self.input_image.data, 0, 1)
        elif self.reg == "gaussian_blur":
            if self.gaussian_count == 0:
                self.input_image.data = GaussianBlur(self.theta_b_width)(
                    self.input_image.data
                )
                self.input_image.data = torch.clamp(self.input_image.data, 0, 1)
            self.gaussian_count = (self.gaussian_count + 1) % self.theta_b_every
        elif self.reg == "none" or self.reg == "clip":
            self.input_image.data = torch.clamp(self.input_image.data, 0, 1)
        elif self.reg == "mix":
            self.input_image.data += (
                -self.theta_decay
                * self.alfa
                * torch.sign(self.input_image.data - mean[None, :, None, None])
            )
            self.input_image.data += (
                -2
                * (self.theta_decay * (1 - self.alfa))
                * (self.input_image.data - mean[None, :, None, None])
            )
            if self.gaussian_count == 0:
                self.input_image.data = GaussianBlur(self.theta_b_width)(
                    self.input_image.data
                )
            self.gaussian_count = (self.gaussian_count + 1) % self.theta_b_every
            self.input_image.data = torch.clamp(self.input_image.data, 0, 1)
        else:
            raise ValueError("Incorrect regularisation.")

    def optimize(self):
        t = trange(self.num_iterations)
        for iteration in t:
            self.backward_pass()
            self.update_input_image()
            if self.reg == "mix" or self.reg == "clip":
                # Clip pixels with small contributions
                contributions = self.compute_pixel_contributions()
                self.clip_pixels_with_small_contributions(
                    contributions, self.theta_c_pct
                )  # Adjust percentile as needed

                # Clip pixels with small norm
                norm_threshold = torch.quantile(
                    self.input_image.data, self.theta_n_pct / 100
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
    if len(argv) != 4:
        print("Usage: python %s <desired_layer> <desired_node> <desire_regularisation>")
        exit(1)

    # from torchvision.models.feature_extraction import get_graph_node_names
    # train_nodes, eval_nodes = get_graph_node_names(self.model)
    # print(train_nodes, eval_nodes)

    reg = argv[3]  # Change the regularisation method if needed
    hidden_layer = argv[1]
    hidden_node = int(argv[2])
    learning_rate = 1
    num_iterations = 200

    # relevant to l1 and l2 and mix
    theta_decay = 0.0001  # Adjust the regularisation strength as needed

    # relevant to gaussian blur and mix
    theta_b_width = 3
    theta_b_every = 1

    # relevant to clip and mix
    theta_n_pct = 0.1
    theta_c_pct = 5

    # relevant to mix
    alfa = 0  # l1 and l2 are implemented in the proportion alfa:1-alfa

    from torchvision.models import alexnet, AlexNet_Weights

    weights = AlexNet_Weights.DEFAULT
    model = alexnet(weights=weights)
    class_names = weights.meta["categories"]

    attack = Visualise(
        model,
        hidden_node,
        reg,
        theta_decay,
        theta_b_width,
        theta_b_every,
        theta_n_pct,
        theta_c_pct,
        learning_rate,
        num_iterations,
        alfa,
        hidden_layer,
    )
    attack.optimize()

    predicted_class, confidence = attack.get_prediction()
    print(f"Image saved at ./layer_{hidden_layer}_node_{hidden_node}_{reg}.png")

    optimized_image = attack.get_optimized_image()
    save_image(
        optimized_image,
        f"layer_{hidden_layer}_node_{hidden_node}_{reg}.png",
        format="png",
    )
