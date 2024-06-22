# 模块注册器
from .registry import Registry


class ModuleRegistry(Registry):
    _type = "module"