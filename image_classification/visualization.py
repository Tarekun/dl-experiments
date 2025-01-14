import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from omegaconf import DictConfig


PLOTS_DIR = "./plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def _format_filename(cfg: DictConfig):
    name = "results"
    name += f"-o{cfg.optimizer._target_}"
    name += f"-lr{cfg.optimizer.lr}"
    name += f"-con{cfg.conv_layers}"
    name += f"-lin{cfg.lin_layers}"
    name += f"-e{cfg.epochs}"

    name = name.replace(".", "_")
    return f"{name}.png"


def _legend_text(cfg: DictConfig):
    legend = ""
    legend += f"optim: {cfg.optimizer._target_}\n"
    legend += f"lr: {cfg.optimizer.lr}\n"
    legend += f"epochs: {cfg.epochs}\n"
    legend += f"conv: {cfg.conv_layers}\n"
    legend += f"lin: {cfg.lin_layers}\n"
    return legend


def create_single_plot(
    filename: str,
    rounds,
    losses,
    accuracies,
    min_accuracy,
    max_accuracy,
    cfg: DictConfig,
):
    fig, ax1 = plt.subplots()

    # Plotting losses
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="red")
    ax1.plot(rounds, losses, color="red")
    ax1.tick_params(axis="y", labelcolor="red")

    # Creating second y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy (%)", color="blue")

    lower_errors = [
        abs(avg - min_acc) for avg, min_acc in zip(accuracies, min_accuracy)
    ]
    upper_errors = [
        abs(max_acc - avg) for avg, max_acc in zip(accuracies, max_accuracy)
    ]
    error = [lower_errors, upper_errors]

    # Plot averaged accuracy with error bars
    ax2.errorbar(
        rounds,
        accuracies,
        yerr=error,
        fmt="o-",
        color="blue",
        ecolor="black",
        capsize=5,
        label="Average Accuracy",
    )

    ax2.tick_params(axis="y", labelcolor="blue")

    ax2.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    plt.title(f"Last Accuracy: {round(accuracies[-1], 2)}%")
    fig.tight_layout()
    legend = _legend_text(cfg)
    fig.text(
        0.5,
        -0.25,
        legend,
        ha="center",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.5),
    )

    plt.grid(True)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def _average_metrics(histories):
    num_histories = len(histories)
    num_rounds = len(histories[0][0])

    loss_sum = [0.0] * num_rounds
    accuracy_sum = [0.0] * num_rounds
    min_accuracy = [float("inf")] * num_rounds
    max_accuracy = [float("-inf")] * num_rounds

    for history in histories:
        loss_list, accuracy_list = history

        # add the values to the corresponding round in the sum lists
        for i in range(num_rounds):
            loss_sum[i] += loss_list[i]
            accuracy_sum[i] += accuracy_list[i]
            # update min and max accuracies
            min_accuracy[i] = min(min_accuracy[i], accuracy_list[i])
            max_accuracy[i] = max(max_accuracy[i], accuracy_list[i])

    # compute the average for each round by dividing the sums by the number of histories
    avg_loss = [loss_sum[i] / num_histories for i in range(num_rounds)]
    avg_accuracy = [accuracy_sum[i] / num_histories for i in range(num_rounds)]
    return avg_loss, avg_accuracy, min_accuracy, max_accuracy


def plot_simulations(
    histories: list[list],
    cfg: DictConfig,
):
    # dir_name = cfg.get("dir_name", PLOTS_DIR)
    dir_name = "."
    losses, accuracies, min_accuracies, max_accuracies = _average_metrics(histories)

    accuracies = [100.0 * value for value in accuracies]
    min_accuracies = [100.0 * value for value in min_accuracies]
    max_accuracies = [100.0 * value for value in max_accuracies]
    rounds = [i for i in range(len(losses))]

    base = os.path.join(PLOTS_DIR, dir_name)
    os.makedirs(base, exist_ok=True)
    filename = _format_filename(cfg)

    create_single_plot(
        f"{base}/{filename}",
        rounds,
        losses,
        accuracies,
        min_accuracies,
        max_accuracies,
        cfg,
    )
