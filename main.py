import torch

from model import NextGoalPredictor
from replay_processing.utils import get_all_bins, get_batch
from torch import nn, optim


def train_loop(model, loss_fn, optimizer, epochs, save_dir, device="cuda", save_frequency=50):
    model.to(device)
    for epoch in range(epochs):
        bins = get_all_bins("replay_processing/bins")
        epoch_loss = 0
        for i, file_names in enumerate(bins):
            batch_arr = get_batch(file_names)
            inputs, labels = batch_arr[0].to(device), batch_arr[1].to(device)
            # Compute prediction and loss
            output = model(inputs)
            loss = loss_fn(output, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"Batch:{i}; Len batch: {len(inputs)}; Loss:{loss.item()}")
            if i % save_frequency == save_frequency - 1:
                torch.save(model.state_dict(), f"{save_dir}/model_{epoch}_{i}.pckl")
        print(f"Epoch {epoch} done with loss of {epoch_loss}")


if __name__ == '__main__':
    model = NextGoalPredictor()
    optimizer = optim.Adam(model.parameters())
    loss_funciton = nn.CrossEntropyLoss(reduction="mean")
    train_loop(model, loss_funciton, optimizer, torch.device("cuda"), epochs=2)
