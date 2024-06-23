import subprocess

class TmuxSession:
    def __init__(self, session_name):
        self.session_name = session_name
        self.create_session()

    def create_session(self):
        """创建一个新的 tmux 会话"""
        try:
            subprocess.run(["tmux", "new-session", "-d", "-s", self.session_name], check=True)
            print(f"Tmux session '{self.session_name}' created.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to create tmux session '{self.session_name}': {e}")

    def run_command(self, command):
        """在 tmux 会话中运行命令"""
        try:
            subprocess.run(["tmux", "send-keys", "-t", self.session_name, command, "C-m"], check=True)
            print(f"Command '{command}' sent to tmux session '{self.session_name}'.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to send command to tmux session '{self.session_name}': {e}")

    def attach(self):
        """附加到 tmux 会话"""
        try:
            subprocess.run(["tmux", "attach-session", "-t", self.session_name], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to attach to tmux session '{self.session_name}': {e}")