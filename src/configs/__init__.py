from hydra.core.config_store import ConfigStore
from . import network, env, train

def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="train_schema", node=train.TrainConfig)