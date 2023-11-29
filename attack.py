from sys import argv, exit
from os.path import splitext
import torch
import torch.optim as optim
import torchvision.transforms as T
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import save_image
from tqdm import trange


class Attack:
    def __init__(
        self,
        model,
        preprocess,
        postprocess,
        input_image,
        required_class,
        reg="l2",
        reg_strength=0.05,
        learning_rate=2,
        num_iterations=100,
    ):
        self.required_class = required_class
        self.reg = reg
        self.reg_strength = reg_strength
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

        # Load the pre-trained model
        self.model = model
        self.model.eval()
        self.preprocess = preprocess
        self.postprocess = postprocess

        # Create a random input image (you can also start with an existing image)
        z = self.preprocess(input_image)
        self.given_input_image = torch.reshape(z, (1, *z.shape))
        self.input_image = self.given_input_image.detach().clone()
        self.input_image.requires_grad = True

        # Create an optimizer
        self.optimizer = optim.SGD([self.input_image], lr=self.learning_rate)

    def forward_pass(self):
        output = self.model(self.input_image)
        activation = output[0, self.required_class]
        return activation

    def backward_pass(self):
        self.model.zero_grad()
        activation = self.forward_pass()
        activation.backward()

    def update_input_image(self):
        self.input_image.data += self.learning_rate * self.input_image.grad.data
        if self.reg == "l2":
            self.input_image.data += (
                -2
                * self.reg_strength
                * (self.input_image.data - self.given_input_image.data)
            )
        elif self.reg == "l1":
            self.input_image.data += -self.reg_strength * torch.sign(
                self.input_image.data - self.given_input_image.data
            )

    def optimize(self):
        t = trange(self.num_iterations)
        for i in t:
            self.backward_pass()
            self.update_input_image()
            self.input_image.grad.zero_()
            t.set_description("Activation: %.2f" % self.forward_pass().item())
            t.refresh()

    def get_prediction(self):
        with torch.no_grad():
            output = self.model(self.input_image)

        predicted_class = output.argmax().item()
        confidence = torch.softmax(output, dim=1)[0, predicted_class].item()
        return predicted_class, confidence

    def get_optimized_image(self):
        return self.postprocess(self.input_image.squeeze(0))


if __name__ == "__main__":
    if len(argv) != 3 and len(argv) != 4:
        print(
            "Usage: python %s <input_image_path> <desired_class_index> [output_image_path]"
        )
        exit(1)

    given_input_image = read_image(argv[1], ImageReadMode.RGB) / 256.0
    required_class = int(argv[2])

    from torchvision.models import alexnet, AlexNet_Weights

    weights = AlexNet_Weights.DEFAULT
    model = alexnet(weights=weights)
    class_names = weights.meta["categories"]

    preprocess = weights.transforms(antialias=None)
    # preprocessing means and std used in torchvision.models
    # required for reconstruction in postprocess
    preprocess_mean = [0.485, 0.456, 0.406]
    preprocess_std = [0.229, 0.224, 0.225]
    postprocess = T.Normalize(
        mean=[-m / s for m, s in zip(preprocess_mean, preprocess_std)],
        std=[1 / s for s in preprocess_std],
    )

    attack = Attack(
        model,
        preprocess,
        postprocess,
        given_input_image,
        required_class,
        reg="l2",
        reg_strength=0.05,
        learning_rate=2,
        num_iterations=10,
    )
    initial_class, initial_confidence = attack.get_prediction()
    print(
        f"Initial Class:   {class_names[initial_class]}, Confidence: {initial_confidence}"
    )

    original_filename, original_filetype = splitext(argv[1])
    original_filetype = original_filetype[1:]

    save_image(
        postprocess(preprocess(given_input_image)),
        "original_" + original_filename + "." + original_filetype,
        format=original_filetype,
    )

    attack.optimize()
    predicted_class, confidence = attack.get_prediction()
    print(f"Desired Class:   {class_names[required_class]}")
    print(f"Predicted Class: {class_names[predicted_class]}, Confidence: {confidence}")

    optimized_image = attack.get_optimized_image()

    if len(argv) != 4:
        output_filename = "adversarial_" + original_filename + "." + original_filetype
        output_filetype = original_filetype
    else:
        output_filename, output_filetype = splitext(argv[3])
        output_filetype = output_filetype[1:]

    save_image(
        optimized_image,
        output_filename + "." + output_filetype,
        format=output_filetype,
    )
