import matplotlib.pyplot as plt


def plot_placeholder_loss():
    """Placeholder function to plot the loss of a model.
    This function is used to demonstrate the use of the visualize module.
    It does not take any input as it is for testing during pipeline setup.
    """
    loss = [0.5, 0.4, 0.3, 0.2, 0.1]
    plt.plot(loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over epochs")
    # plt.show()
    plt.savefig("reports/figures/Loss_plot.png")

if __name__ == "__main__":
    plot_placeholder_loss()