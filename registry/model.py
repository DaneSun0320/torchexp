# 模块注册器
from .registry import Registry


class ModelRegistry(Registry):
    _type = "model"