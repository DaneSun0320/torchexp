import torch
import torch.nn as nn
from registry import ModuleRegistry


@ModuleRegistry.register("attention", name="SelfAttention")
class SelfAttention(nn.Module):
    pass
