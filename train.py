import argparse
from pathlib import Path

try:
    from tqdm.rich import tqdm  # nice progress bar
except ImportError:
    from tqdm import tqdm

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml_template.models import MLP
from ml_template.utils import (
    DotDict,
    learning_rate_schedule,
    seed_everything,
    set_learning_rate,
    to_numpy,
)
from ml_template.wandb_utils import DummyWanb


def load_data_and_model(config):
    X, Y = torch.randn(1000, 1), torch.randn(1000, 1)
    trainset, valset, testset = TensorDataset(X, Y), TensorDataset(X, Y), None
    trainloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=config.batch_size, shuffle=False)
    testloader = None

    model = MLP(input_dim=1, hidden_dims=(config.hidden_dim,), output_dim=1)
    return (trainset, valset, testset), (trainloader, valloader, testloader), model


def loss_fn(model, batch):
    x, y = batch
    loss = (model(x) - y).pow(2).mean(0)
    return loss


def plot_arrays(x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    return fig


def eval_on_batch(model, batch):
    x, y = batch

    metrics, figs = {}, {}

    with torch.no_grad():
        loss = loss_fn(model, batch)
        metrics["loss"] = loss.item()

    # add more plotting functions here
    # plotting functions should return a matplotlib Figure
    fig = plot_arrays(to_numpy(x), to_numpy(y))
    figs["prediction"] = fig

    return metrics, figs


def run(config: DotDict, run_dir: Path, wandb_run=DummyWanb.init()):
    seed_everything(config.seed)

    #### Load data
    _, (trainloader, valloader, _), model = load_data_and_model(config)

    model = model.to(config.device)
    # model = torch.jit.script(model) # optional but will probably speed up by at least 2x; can also try torch.jit.trace
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # wandb_run.watch(model, **config.watch)  # log gradients of model, not compatible with ScriptModules
    num_batches_per_epoch = len(trainloader)  # won't work for iter type dataloaders
    trainloader_iter = iter(trainloader)
    valloader_iter = iter(valloader)

    fixed_train_samples = next(trainloader_iter)  # fix some samples for metrics
    fixed_val_samples = next(valloader_iter)  # only use one batch for validation
    fixed_samples = dict(train=fixed_train_samples, val=fixed_val_samples)

    if config.train:
        for after_i_updates in tqdm(range(config.num_updates + 1)):
            epoch = 1 + after_i_updates // num_batches_per_epoch
            wandb_run.log({"epoch": epoch}, commit=False)

            # loop through batches indefinitely
            try:
                batch = next(trainloader_iter)  # may be good to add a batch_idx
            except StopIteration:
                trainloader_iter = iter(trainloader)
                batch = next(trainloader_iter)
            x, y = [_.to(config.device, non_blocking=True).contiguous() for _ in batch]
            batch = (x, y)  # TODO: make into a dataclass?

            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            wandb_run.log({"train/loss": loss.item()}, commit=False)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                model.parameters(), max_norm=1e2, error_if_nonfinite=True
            )
            wandb_run.log({"train/grad_norm": grad_norm.item()}, commit=False)

            lr = learning_rate_schedule(after_i_updates + 1, config.lr_warmup_steps, config.lr, config.num_updates)
            set_learning_rate(optimizer, lr)
            wandb_run.log({"lr": lr}, commit=False)

            optimizer.step()

            # Training metrics and figures
            model.eval()
            if after_i_updates % config.eval_every_n_steps == 0:
                for phase in ["train", "val"]:
                    metrics, figs = eval_on_batch(
                        model,
                        fixed_samples[phase],
                    )
                    metrics = {f"{phase}/{k}": v for k, v in metrics.items()}
                    figs = {f"{phase}/{k}": v for k, v in figs.items()}

                    wandb_run.log(metrics, commit=False)
                    wandb_run.log(figs, commit=False)

                    [plt.close(f) for f in figs.values()]

                model_path = run_dir / "model.pt"
                torch.save(model.state_dict(), model_path)
                wandb_run.save(str(model_path))
                print(f"Saved model after {after_i_updates} updates")

            wandb_run.log({"after_gradient_updates": after_i_updates}, commit=True)  # finally commit it all at the end

    # TODO: test phase
    if config.evaluate:
        pass


def main(**config) -> None:
    # Instantiate config
    config = DotDict(config)

    if config.debug:  # settings for quick development run
        wandb = DummyWanb
        config.seed = 0
        # config.num_updates = 1
        # config.eval_every_n_steps = 1
    else:
        import wandb

    #### Set up logger
    tags = config.tags.split()  # assume "tag1 tag2 tag3"
    wandb_run = wandb.init(
        project=config.project, notes=config.notes, tags=tags, config=config, name=config.name, group=config.group
    )
    run_dir = Path(wandb_run.dir).resolve()
    wandb_run.config.update(dict(run_dir=str(run_dir)))

    #### Train and evaluate
    _ = run(config, run_dir, wandb_run)

    #### Close resources
    wandb_run.finish()
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # misc stuff
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", default=17, type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    # logger stuff
    parser.add_argument("--project", default="ml_template", type=str)
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--notes", default="", type=str)
    parser.add_argument("--tags", default="", type=str)  # "tag1 tag2 tag3"
    parser.add_argument("--group", default="", type=str)
    # data stuff
    parser.add_argument("--num_updates", default=1000, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    # model stuff
    parser.add_argument("--hidden_dim", default=64, type=int)
    # optimizer stuff
    parser.add_argument("--lr", default=2e-1, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--lr_warmup_steps", default=10, type=int)
    # training stuff
    parser.add_argument("--eval_every_n_steps", default=100, type=int)
    FLAGS = parser.parse_args()

    main(**FLAGS.__dict__)
