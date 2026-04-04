from typing import Dict, List, Any
from operator import attrgetter
import torch


def build_optimizers_parameters(model: torch.nn.Module, cfg) -> List[Dict[str, Any]]:
    params = []
    for module_path, group_settings in cfg.items():
        module = attrgetter(module_path)(model)
        
        param_group = {'params': module.parameters()}
        param_group.update(dict(group_settings))
        params.append(param_group)

    return params