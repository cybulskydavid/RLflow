import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig):
    print(cfg)