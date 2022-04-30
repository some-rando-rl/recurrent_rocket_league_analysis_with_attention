import torch
import wandb
from model import NextGoalPredictor
from dataset import RocketLeagueReplayDataset
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from os.path import join as p_join

config = {
    "seed": 0,
    "batch_size": 10,
    "lr": 5e-5,
    "save_frequency": 100,
    "eval_frequency": 100,
    "device": "cuda"
}

torch.manual_seed(config["seed"])

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    xx_pad = pad_sequence(xx, batch_first=True)
    yy_pad = pad_sequence(yy, batch_first=True)

    return xx_pad, yy_pad

def train_loop(model, loss_fn, optimizer, save_dir, epochs=1, batch_size=config["batch_size"], device=torch.device(config["device"]), save_frequency=config["save_frequency"], eval_frequency=config["eval_frequency"]):
    dataloader = DataLoader(
        RocketLeagueReplayDataset(),
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        collate_fn=pad_collate
    )

    model.to(device)
    parameter_vector = nn.utils.parameters_to_vector(model.parameters())

    print(f"Training model with {len(parameter_vector)} params")

    for epoch in range(epochs):
        epoch_loss = 0

        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = torch.tensor([[a[0]] for a in labels]).to(device)

            # zero gradients for every batch
            optimizer.zero_grad()

            # Compute prediction and loss
            output = model(inputs)
            loss = loss_fn(output, labels)
            loss.backward()

            # backpropagate
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
                val_loss, val_acc = validation_run(model, loss_fn, batch_size, device)
                print(f"Validation loss: {val_loss}; Validation accuracy: {val_acc}")
                wandb.log({"val_loss":val_loss, "val_accuracy": val_acc})

        print(f"Epoch {epoch} done with loss of {epoch_loss}")

def validation_run(model, loss_fn, batch_size, device, limit=50):
    dataloader = DataLoader(
        RocketLeagueReplayDataset(root_dir=p_join("data","validation")),
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        collate_fn=pad_collate
    )

    model.to(device)
    with torch.no_grad():
        validation_loss, correct, all = 0, 0, 0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = torch.tensor([[a[0]] for a in labels]).to(device)
            # Compute prediction and loss
            output = model(inputs)
            validation_loss += loss_fn(output, labels).item()
            correct += (output.argmax(1) == labels).type(torch.float).sum().item()
            all += len(labels)
            if limit and i >= limit:
                break
    return validation_loss, correct / all


def main():
    wandb.init(project="goal-prediction-carrot", entity="some-rando-rl", config=config)
    model = NextGoalPredictor()

    #load = input("Load [Y/n]: ").lower()
    #if load == "l" or load == "":
    #    model.load_state_dict(torch.load("models/model2_2_499"))

    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    loss_function = nn.CrossEntropyLoss()

    train_loop(
        model=model,
        loss_fn=loss_function,
        optimizer=optimizer,
        save_dir="models",
        epochs=10,
        device=torch.device("cuda")
    )


if __name__ == '__main__':
    main()
