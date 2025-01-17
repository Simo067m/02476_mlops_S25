import matplotlib.pyplot as plt


def plot_loss(train_vals: list, val_vals: list, test_vals: list, save_path: str) -> None:
    """Plots the loss for the specific metric: Train, Validation or Test.
    Saves the plot to the specified path.
    """
    plt.figure()
    plt.plot(train_vals, label="Train")
    plt.plot(val_vals, label="Validation")
    plt.plot(test_vals, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Model Loss")
    plt.legend()
    plt.savefig(save_path + "/Loss_plot.png")

def plot_accuracy(accuracy_vals: list, save_path: str) -> None:
    """Plots the accuracy for the specific metric: Train, Validation or Test.
    Saves the plot to the specified path.
    """
    plt.figure()
    plt.plot(accuracy_vals, label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Model Accuracy")
    plt.legend()
    plt.savefig(save_path + "/Accuracy_plot.png")

if __name__ == "__main__":
    metric = "Train"
    metric_values = [0.5, 0.4, 0.3, 0.2, 0.1]
    save_path = "reports/figures"
    plot_loss(metric, metric_values, save_path)