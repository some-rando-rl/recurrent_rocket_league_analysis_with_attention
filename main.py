import torch
import wandb
from model import NextGoalPredictor
from replay_processing.utils import get_all_bins, get_batch
from torch import nn, optim


def train_loop(model, loss_fn, optimizer, save_dir, epochs=1, device=torch.device("cpu"), save_frequency=50, eval_frequency=50):
    model.to(device)
    parameter_vector = nn.utils.parameters_to_vector(model.parameters())
    for epoch in range(epochs):
        bins = get_all_bins("replay_processing/bins", 900)
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

            new_parameter_vector = nn.utils.parameters_to_vector(model.parameters())
            update_magnitude = torch.linalg.vector_norm(parameter_vector-new_parameter_vector)
            wandb.log({"update_magnitude":update_magnitude, "training_loss":loss.item()})
            parameter_vector = new_parameter_vector

            epoch_loss += loss.item()
            print(f"Batch:{i}; Len batch: {len(inputs)}; Loss:{loss.item()}; Update magnitude: {update_magnitude}")
            if i % save_frequency == save_frequency - 1:
                torch.save(model.state_dict(), f"{save_dir}/model_{epoch}_{i}.pckl")
            if i % eval_frequency == eval_frequency- 1:
                val_loss, val_acc = validation_run(model, loss_fn, device)
                print(f"Validation loss: {val_loss}; Validation accuracy: {val_acc}")
                wandb.log({"val_loss":val_loss, "val_accuracy": val_acc})
        print(f"Epoch {epoch} done with loss of {epoch_loss}")


def validation_run(model, loss_fn, device):
    model.to(device)
    with torch.no_grad():
        validation_loss, correct, all = 0, 0, 0
        bins = get_all_bins("replay_processing/validation_bins", 90)
        for i, file_names in enumerate(bins):
            batch_arr = get_batch(file_names)
            inputs, labels = batch_arr[0].to(device), batch_arr[1].to(device)
            # Compute prediction and loss
            output = model(inputs)
            validation_loss += loss_fn(output, labels).item()
            correct += (output.argmax(1) == labels).type(torch.float).sum().item()
            all += len(labels)
    return validation_loss, correct / all


def main():
    wandb.init(project="next-goal-predictor", entity="mrkvicka02")
    model = NextGoalPredictor()
    if input("[L]oad").lower() == "l":
        model.load_state_dict(torch.load("models/model2_2_499"))
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    loss_funciton = nn.CrossEntropyLoss(reduction="mean")
    train_loop(model, loss_funciton, optimizer, "models", epochs=10, device=torch.device("cuda"))


if __name__ == '__main__':
    main()
