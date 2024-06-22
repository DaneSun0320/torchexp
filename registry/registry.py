"""
    注册器基类，用于注册模块、损失函数等实验相关的组件
    各个类型的注册器继承自BaseRegistry，通过修改__type属性来实现不同类型的注册器
    通过register装饰器注册组件，通过get_module方法获取组件
    示例：
    # 损失函数注册类
    class CriterionRegistry(Registry):
        __type = "criterion"
    # 模型模块注册类
    class ModuleRegistry(Registry):
        __type = "module"
    # 日志记录器注册类
    class LoggerRegistry(Registry):
        __type = "logger"
    # 使用
    @CriterionRegistry.register("criterion1")
    class Criterion1:
        pass
    @ModuleRegistry.register("head", "Head_Segmentation")
    class HeadSegmentation:
        pass
    # 获取
    criterion = Registry.get_module("criterion", "criterion1")
    module = ModuleRegistry.get_module("head", "Head_Segmentation")
    # 打印注册树
    BaseRegistry.show_modules_tree()

"""

from typing import Union, List

# 注册器基类
class _BaseRegistry:
    _MODULE_DICT = dict()
    __type = "root"

    @classmethod
    def _add_cls_to_dict(cls, keys, value):
        if isinstance(keys, str):
            keys = [keys]
        if not isinstance(keys, list):
            raise TypeError("keys must be a list or a string")
        current_dict = cls._MODULE_DICT
        for key in keys[:-1]:
            current_dict = current_dict.setdefault(key, {})
        # 判断cls_name 在同一parent中是否已经存在
        if keys[-1] in cls._get_leaf_dict(keys[:-1]):
            raise ValueError(f"Name `{keys[-1]}` already exists in `{'-> '.join(keys)}`")
        if value is not None:
            current_dict.setdefault(keys[-1], []).append(value)

    @classmethod
    def _get_leaf_dict(cls, parent):
        """
        根据键链列表访问嵌套字典中的值。

        参数：
        d (dict): 目标字典
        key_chain (list): 键链列表

        返回：
        目标值，或KeyError如果键链不正确
        """
        current = cls._MODULE_DICT
        for key in parent:
            if not isinstance(current, dict):
                raise KeyError(f"parent does not lead to a valid dictionary. Current segment: {key}")
            if key not in current:
                raise KeyError(
                    f"Module `{key}` not found in the registry tree. You can use `show_modules_tree` to check the registry tree")
            current = current[key]
        return current

    @classmethod
    def register(cls, parent: Union[List, str], name: str = None, allow_instance=True):
        def decorator(module_cls):
            cls_name = name
            if name is None:
                # 获取类名
                cls_name = module_cls.__name__
            # 将cls存入self.MODULE_DICT[parent]中
            if not isinstance(parent, list):
                cls._add_cls_to_dict([cls.__type, parent, cls_name], module_cls)
            else:
                parent.insert(0, cls.__type)
                parent.append(cls_name)
                cls._add_cls_to_dict(parent, module_cls)
            if isinstance(module_cls, type):
                # 通过装饰器返回一个新的类，防止直接实例化
                if not allow_instance:
                    class Module(module_cls):
                        def __new__(cls, *args, **kwargs):
                            raise TypeError("Module must be obtained through the get_module method")
                else:
                    Module = module_cls

                return Module
            else:
                # 如果module_cls是函数，返回一个装饰器，用于实例化
                def wrapper(*args, **kwargs):
                    return module_cls(*args, **kwargs)

                return wrapper

        return decorator

    @classmethod
    def get_module(cls, *module_path, module=None):
        module = module if module else cls._MODULE_DICT["root"]
        for module_name in module_path:
            if module_name not in module:
                raise KeyError(f"{module_name} not exists in the registry")
            module = module[module_name]
        return module

    @classmethod
    def show_modules_tree(cls):
        # 递归打印self.MODULE_DICT
        def print_tree(module_dict, depth=0):
            for name, module in module_dict.items():
                print("    " * depth + f"|-- {name}")
                if isinstance(module, dict):
                    print_tree(module, depth + 1)

        print_tree(cls._MODULE_DICT)
        print("=" * 50)

    @classmethod
    def get_all(cls):
        return cls._MODULE_DICT


# 组件注册器
class Registry(_BaseRegistry):
    _MODULE_DICT = _BaseRegistry.get_all()
    _type = None

    @classmethod
    def register(cls, *args, name: str = None, allow_instance=True):
        args = list(args)
        parent = [cls._type] + args if cls._type is not None else args
        return super().register(parent, name=name, allow_instance=allow_instance)

    @staticmethod
    def _is_leaf_node(d, key_chain):
        """
        判断键链是否访问字典中的叶子节点。

        参数：
        d (dict): 目标字典
        key_chain (list): 键链列表

        返回：
        bool: 如果是叶子节点，返回True；否则返回False
        """
        current = d
        for key in key_chain:
            if not isinstance(current, dict) or key not in current:
                return False  # 当前不是字典，或者键不存在
            current = current[key]

        # 检查最后一个键对应的值是否为字典或列表
        return not isinstance(current, (dict, list))

    @classmethod
    def get_module(cls, *module_path):
        assert "root" in cls._MODULE_DICT, "Please register the module first!"
        # 判断是否是叶子节点
        assert cls._is_leaf_node(cls._MODULE_DICT, module_path), "Please specify the module type!"
        module = cls._MODULE_DICT["root"] if cls._type is None else cls._MODULE_DICT["root"][cls._type]
        return super().get_module(*module_path, module=module)


if __name__ == "__main__":
    @Registry.register("head")
    class HeadSegmentation:
        pass


    @Registry.register("criterion")
    class AdaptiveLoss:
        pass


    @Registry.register("criterion")
    class CrossEntropyLoss:
        pass


    @Registry.register("module")
    class ResNet34:
        pass


    Registry.show_modules_tree()
