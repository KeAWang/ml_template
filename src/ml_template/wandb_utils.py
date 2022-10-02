import os
import pickle
from pathlib import Path

import wandb
from wandb.errors import CommError


def get_history(user="", project="", query={}, **kwargs):
    api = wandb.Api()
    runs = api.runs(path=f"{user}/{project}", filters=query)
    dataframes = [run.history(**kwargs) for run in runs]
    return list(zip(runs, dataframes))


def download_files(user="", project="", query={}, save_dir=".", specific_files=None, overwrite=False, **kwargs):
    """
    Download the files of each run into a new directory for the run.
    Also saves the config dict of the run.
    See https://docs.wandb.com/library/reference/wandb_api for how to write queries
    """
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    if specific_files is not None:
        specific_files = set(specific_files)

    api = wandb.Api()
    runs = api.runs(path=f"{user}/{project}", filters=query)
    run_dirs = []
    for run in runs:
        name = run.name
        config = run.config

        run_dir = os.path.join(save_dir, name)
        run_dirs.append(Path(run_dir))
        if not os.path.isdir(run_dir):
            os.mkdir(run_dir)

        with open(os.path.join(run_dir, "config.pkl"), "wb") as h:
            pickle.dump(config, h)

        files = run.files()
        for file in files:
            if specific_files is None or file.name in specific_files:
                try:
                    file.download(root=run_dir, replace=overwrite)
                except CommError:
                    print(
                        f"Failed to download {file.name} for run {name}. File already exists! Set overwrite=True to overwrite."
                    )

    return runs, run_dirs


def get_config(user="", project="", query={}):
    api = wandb.Api()
    runs = api.runs(path=f"{user}/{project}", filters=query)

    configs = [(run.name, run.config) for run in runs]
    return configs


def config_to_omegaconf(config: dict):
    from omegaconf import OmegaConf

    keys, values = zip(*config.items())

    # convert from keys that look like "datamodules/batch_size" into "datamodules.batch_size"
    dot_keys = [key.replace("/", ".") for key in keys]

    # convert "None" strings into "null" for OmegaConf to parse it as a None object
    new_values = ["null" if v == "None" else v for v in values]

    dot_list = [f"{k}={v}" for k, v in zip(dot_keys, new_values)]
    omega_conf = OmegaConf.from_dotlist(dot_list)
    return omega_conf


class DummyWanb:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.config = {}
        self.name = ""
        self.id = ""
        self.path = ""
        self.dir = "./"

    @staticmethod
    def init(*args, **kwargs):
        return DummyWanb(*args, **kwargs)

    def log(self, *args, **kwargs):
        return

    def watch(self, *args, **kwargs):
        return

    def finish(self, *args, **kwargs):
        return

    def save(self, *args, **kwargs):
        return
