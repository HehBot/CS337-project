from visualise_output_neuron import Visualise


def grid_search(
    model,
    class_idx,
    reg,
    theta_decay,
    theta_b_width,
    theta_n_pct,
    theta_c_pct,
    learning_rate,
    num_iterations,
    alfa,
):
    from torchvision.utils import save_image

    for c_pct in theta_c_pct:
        for b_width in theta_b_width:
            for n_pct in theta_n_pct:
                for a in alfa:
                    vis = Visualise(
                        model,
                        class_idx,
                        reg,
                        theta_decay,
                        b_width,
                        1,
                        n_pct,
                        c_pct,
                        learning_rate,
                        num_iterations,
                        a,
                    )
                    vis.optimize()
                    optimized_image = vis.get_optimized_image()
                    #                     predicted, conf = vis.get_prediction()
                    save_image(
                        optimized_image,
                        "figures/"
                        + f"alpha_{a}_n_pct_{n_pct}_blur_size_{b_width}_c_pct{c_pct}.png",
                        format="png",
                    )


if __name__ == "__main__":
    class_idx = 366  # Replace with the index of your target class
    reg = "mix"  # Change the regularization method if needed

    from torchvision.models import alexnet, AlexNet_Weights

    weights = AlexNet_Weights.DEFAULT
    model = alexnet(weights=weights)
    class_names = weights.meta["categories"]

    # grid search parameters:
    theta_decay = 0.01  # Adjust the regularization strength as needed
    theta_b_width = [1, 3, 5]
    theta_n_pct = [0, 0.1, 0.2]
    theta_c_pct = [0, 5, 10]
    alfa = [0, 0.2, 0.4]
    learning_rate = 1
    num_iterations = 200

    grid_search(
        model,
        class_idx,
        reg,
        theta_decay,
        theta_b_width,
        theta_n_pct,
        theta_c_pct,
        learning_rate,
        num_iterations,
        alfa,
    )
