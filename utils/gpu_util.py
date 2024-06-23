import time
import psutil
from pynvml import *

# 初始化 NVML
nvmlInit()

# 获取 GPU 数量
device_count = nvmlDeviceGetCount()

def get_gpu_processes(gpu_index=0):
    try:
        handle = nvmlDeviceGetHandleByIndex(gpu_index)
        processes = nvmlDeviceGetComputeRunningProcesses(handle)
        return processes
    except NVMLError as err:
        print(f"Failed to get GPU processes: {err}")
        return []

def gpu_is_available(gpu_index=0):
    processes = get_gpu_processes(gpu_index)
    if not processes:
        return True, None
    else:
        pids = [p.pid for p in processes]
        return False, pids

def monitor_gpu_processes(gpu_index=0, interval=5):
    print(f"Monitoring GPU {gpu_index} processes...")
    while True:
        available, pids = gpu_is_available(gpu_index)
        if available:
            print("GPU is available.")
        else:
            print(f"GPU is being used by processes: {pids}")
            for pid in pids:
                process_info = get_process_info(pid)
                if process_info:
                    name, cpu_percent, memory = process_info
                    print(f"PID: {pid}, Name: {name}, CPU: {cpu_percent}%, Memory: {memory / 1024 ** 2:.2f}MB")

        time.sleep(interval)


if __name__ == "__main__":
    monitor_gpu_processes(1)