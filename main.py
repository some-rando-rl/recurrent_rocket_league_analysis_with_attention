import torch
import wandb
from model import NextGoalPredictor
from dataset import RocketLeagueReplayDataset
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from os.path import join as p_join

#run_id = "10tvfyjw"
run_id = None
config = {
    "seed": 0,
    "batch_size": 10,
    #"lr": 1e-6,
    "save_frequency": 3,
    "eval_frequency": 3,
    "device": "cuda",
}

torch.manual_seed(config["seed"])

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    xx_pad = pad_sequence(xx, batch_first=True)
    yy_pad = pad_sequence(yy, batch_first=True)

    return xx_pad, yy_pad

def train_loop(model, save_dir, epochs=1, batch_size=config["batch_size"], device=torch.device(config["device"]), save_frequency=config["save_frequency"], eval_frequency=config["eval_frequency"]):
    dataloader = DataLoader(
        RocketLeagueReplayDataset(),
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        collate_fn=pad_collate
    )

    loss_fn = nn.CrossEntropyLoss()

    model.to(device)
    parameter_vector = nn.utils.parameters_to_vector(model.parameters())

    print(f"Training model with {len(parameter_vector)} params")

    for epoch in range(epochs):
        if epoch < 50:
            lr = 5e-5
        if 50 <= epoch < 250:
            lr = 1e-5
        if 250 <= epoch < 500:
            lr = 5e-6
        if 500 <= epoch:
            lr = 1e-6

        optimizer = optim.Adam(model.parameters(), lr)
        epoch_loss = 0

        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            #labels = torch.tensor([[a[0]] for a in labels]).to(device)

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
            labels = labels.to(device)
            # Compute prediction and loss
            output = model(inputs)
            validation_loss += loss_fn(output, labels).item()

            correct += (((output > 0.5).float() - labels) == 0).float().sum()

            #correct += (output > 0.5).type(torch.float) == labels.type(torch.float).sum().item()

            all += labels.numel()
            if limit and i >= limit:
                break
    return validation_loss, correct / all


def main():
    wandb.init(
        project="goal-prediction-carrot",
        entity="some_rando_rl",
        config=config,
        id=run_id
    )
    model = NextGoalPredictor()

    #load = input("Load [Y/n]: ").lower()
    #if load == "l" or load == "":
    #model.load_state_dict(torch.load("models/model_113_2.pckl"))

    train_loop(
        model=model,
        save_dir="models",
        epochs=10000,
        device=torch.device("cuda")
    )


if __name__ == '__main__':
    main()
