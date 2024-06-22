import argparse

from model import auto_import_modules
from registry import Registry
from utils.config_paraser import ConfigParser

def main(args):
    # 初始化模块
    auto_import_modules()
    # 显示模块树
    Registry.show_modules_tree()
    # 加载配置文件
    config = ConfigParser(args.config)

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Distributed training job')
        # 配置文件路径
        parser.add_argument('-c', '--config', type=str, required=True, help='Config file name')
        # 训练轮数
        parser.add_argument('-e', '--epochs', type=int, default=300, help='Total epochs to train the model.py')
        # batch size
        parser.add_argument('-b', '--batch_size', default=32, type=int,
                            help='Input batch size on each device (default: 32)')
        # 验证间隔
        parser.add_argument('--val_interval', type=int, default=10, help='How often to validate')
        # 保存间隔
        parser.add_argument('--save_interval', type=int, default=10, help='How often to save a checkpoint')
        # 初始学习率
        parser.add_argument('--start_lr', default=5e-4, type=float, help='Initial learning rate (default: 1e-4)')
        # 开启混合精度训练
        parser.add_argument('--amp', action='store_true', help='Use mixed precision training')
        # 预热比例
        parser.add_argument('--warmup_ratio', type=float, default=0)
        # 严格加载模型权重
        parser.add_argument('--no-strict', action='store_true', default=False,
                            help='Strict loading of model.py weights')
        # 随机种子
        parser.add_argument('--seed', default=42, type=int, help='Random seed (default: 0)')
        # 跳过层
        parser.add_argument('--skip', nargs='*', default=None)
        weight_arg_group = parser.add_mutually_exclusive_group()
        # 恢复训练
        weight_arg_group.add_argument('--resume', action='store_true', help='Resume training from a checkpoint')
        # 加载权重
        weight_arg_group.add_argument('--weight', type=str, default=None, help='Path to a checkpoint to load')
        freeze_arg_group = parser.add_mutually_exclusive_group()
        # 冻结层
        freeze_arg_group.add_argument('--freeze', nargs='*', default=None)
        # 冻结指定层以外的层
        freeze_arg_group.add_argument('--freeze_except', nargs='*', default=None)
        args = parser.parse_args()
        main(args)
