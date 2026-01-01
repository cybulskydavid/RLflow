from hydra.core.config_store import ConfigStore
from . import network

def register_config():
    cs = ConfigStore.instance()
