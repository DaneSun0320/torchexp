import argparse
from utils.config_paraser import ConfigParser
from utils.session import TmuxSession


def main(args):
    task = ConfigParser(args.task).get_config()
    session = TmuxSession("task")
    # 切换到虚拟环境
    session.run_command(f"conda activate {task.conda_env}")
    # 运行任务
    for t in task.tasks:
        session.run_command(t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tasks')
    # 位置参数
    parser.add_argument('task', type=str, default="./tasks.yaml",help='Task config path')
    args = parser.parse_args()
    main(args)

