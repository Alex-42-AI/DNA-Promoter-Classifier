from math import inf

from pathlib import Path

from copy import deepcopy

from matplotlib import use

from numpy import array, transpose

from random import sample, randint, seed

from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader

from torch import nn, tensor, float32, long, optim, no_grad, argmax, cuda, manual_seed

use("Agg")

from matplotlib import pyplot as plt


def load():
    p, n = [], []

    with open("data.txt", 'r') as f:
        for line in f:
            sequence, flag = line.upper().strip().split()
            sequence = sequence[START:END]

            if int(flag):
                p.append(sequence)

            else:
                n.append(sequence)

    return p, n


def encode(sequence):
    return array([mapping[x] if x in mapping else array([0, 0, 0, 0]) for x in sequence])


mapping = {"A": array([1, 0, 0, 0]),
           "C": array([0, 1, 0, 0]),
           "G": array([0, 0, 1, 0]),
           "T": array([0, 0, 0, 1])}

START, END = 0, 2000

LR = 5e-4

EPOCHS = 15

CHANNELS_1, CHANNELS_2, CHANNELS_3, CHANNELS_4, FC_SIZE = 8, 16, 32, 64, 128

KER_SIZE, BATCH_SIZE = 15, 32

POSITIVES, NEGATIVES = 5920, 5920

OPTIMIZER = optim.Adam

LOSS = nn.CrossEntropyLoss

TRAIN, VALIDATE, TEST = 70, 15, 15

CASE = 3

EPOCHS_len = len(str(EPOCHS))

results_dir = Path(f"results/case{CASE:03d}")

results_dir.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = results_dir / "results.txt"

METADATA_PATH = results_dir / "metadata.txt"

for RUN in range(1, 11):
    SEED = randint(0, 10 ** 6)
    seed(SEED)
    manual_seed(SEED)
    cuda.manual_seed(SEED)
    cuda.manual_seed_all(SEED)

    model = nn.Sequential(
        nn.Conv1d(4, CHANNELS_1, kernel_size=KER_SIZE),
        nn.BatchNorm1d(CHANNELS_1),
        nn.ReLU(),
        nn.MaxPool1d(2),

        nn.Conv1d(CHANNELS_1, CHANNELS_2, kernel_size=KER_SIZE),
        nn.BatchNorm1d(CHANNELS_2),
        nn.ReLU(),
        nn.MaxPool1d(2),

        nn.Conv1d(CHANNELS_2, CHANNELS_3, kernel_size=KER_SIZE),
        nn.BatchNorm1d(CHANNELS_3),
        nn.ReLU(),
        nn.MaxPool1d(2),

        nn.Conv1d(CHANNELS_3, CHANNELS_4, kernel_size=KER_SIZE),
        nn.BatchNorm1d(CHANNELS_4),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.AdaptiveMaxPool1d(1),

        nn.Flatten(),

        nn.Linear(CHANNELS_4, FC_SIZE),
        nn.ReLU(),

        nn.Linear(FC_SIZE, 2)
    )
    device = "cuda" if cuda.is_available() else "cpu"

    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    model.train()

    positive, negative = load()
    positive, negative = sample(positive, POSITIVES), sample(negative, NEGATIVES)

    X = array([encode(seq) for seq in positive + negative])
    X = transpose(X, (0, 2, 1))

    y = array([1] * POSITIVES + [0] * NEGATIVES)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(VALIDATE + TEST) / 100,
        stratify=y,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=TEST / (100 - TRAIN),
        stratify=y_temp,
        random_state=42
    )

    X_train_tensor = tensor(X_train, dtype=float32)
    y_train_tensor = tensor(y_train, dtype=long)

    X_val_tensor = tensor(X_val, dtype=float32).to(device)
    y_val_tensor = tensor(y_val, dtype=long).to(device)

    X_test_tensor = tensor(X_test, dtype=float32).to(device)
    y_test_tensor = tensor(y_test, dtype=long).to(device)

    criterion = LOSS()
    optimizer = OPTIMIZER(model.parameters(), lr=LR)

    if RUN > 1:
        with open(METADATA_PATH, "a") as f:
            f.write(f"Run {RUN}: seed = {SEED}\n")

    else:
        metadata = f"""device = {device}
positives = {POSITIVES}
negatives = {NEGATIVES}
sequence[{START}:{END}]
split = {TRAIN}/{VALIDATE}/{TEST}
model = {model}
parameters = {params}
lr = {LR}
batch_size = {BATCH_SIZE}
epochs = {EPOCHS}
loss = {LOSS.__name__}
optimizer = {OPTIMIZER.__name__}
Run {RUN}: seed = {SEED}
"""

        with open(METADATA_PATH, "w") as f:
            f.write(metadata)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    epochs_range = range(EPOCHS)
    loader_len = len(train_loader)
    best_epoch, best_val, best_state = 0, inf, None
    train_loss, validation_loss, val_accuracy = [], [], []
    result = f"Run {RUN}:\n"

    for epoch in epochs_range:
        print(epoch)
        total_loss = 0

        model.train()

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)

            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        total_loss /= loader_len

        model.eval()

        with no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            val_preds = argmax(val_outputs, dim=1)
            val_acc = (val_preds == y_val_tensor).float().mean()

            if val_loss.item() < best_val:
                best_epoch = epoch
                best_val = val_loss.item()
                best_state = deepcopy(model.state_dict())

        train_loss.append(total_loss)
        validation_loss.append(val_loss.item())
        val_accuracy.append(val_acc.item())
        result += f"{epoch:0{EPOCHS_len}d}; train_loss = {total_loss:.4f}; val_loss = {val_loss.item():.4f}; val_acc = {val_acc.item():.4f}\n"

    model.load_state_dict(best_state)
    model.eval()

    with no_grad():
        outputs = model(X_test_tensor)
        preds = argmax(outputs, dim=1)
        acc = (preds == y_test_tensor).float().mean()
        result += f"test accuracy: {acc.item():.4f}; best epoch: {best_epoch}\n"
        print(acc.item())

    with open(RESULTS_PATH, "a+") as f:
        f.write(result)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(epochs_range, train_loss, label="Train Loss", color="red")
    ax1.plot(epochs_range, validation_loss, label="Validation Loss", color="green")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2 = ax1.twinx()
    ax2.plot(epochs_range, val_accuracy, label="Validation Accuracy", color="blue")

    ax2.set_ylabel("Accuracy")

    ax1.axvline(best_epoch, linestyle="--")

    plt.title(f"Test Accuracy = {acc.item():.4f} | Best Epoch = {best_epoch}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")

    plot_path = f"results/case{CASE:03d}/test_run{RUN:03d}.png"

    plt.savefig(plot_path, bbox_inches="tight")

    plt.close(fig)
