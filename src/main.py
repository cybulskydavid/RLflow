import torch
import random

import numpy as np
import hydra


def set_glogal_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="configs", config_name="train")
def main(cfg):
    set_glogal_seed(cfg.seed)


if __name__ == "__main__":
    main()