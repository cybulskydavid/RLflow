from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)


    def log_params(self, params: DictConfig):
        config_str = OmegaConf.to_yaml(params)
        text_formatted = f"```yaml\n{config_str}\n```"
        self.writer.add_text("Hyperparameters", text_formatted, 0)


    def log_metrics(self, metrics: dict, step: int):
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
            

    def close(self):
        self.writer.close()