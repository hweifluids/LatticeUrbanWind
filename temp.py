import subprocess

cmd = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]
total_mem_mib = int(subprocess.check_output(cmd).decode().split()[0])
print(f"GPU0 总显存：{total_mem_mib} MiB")
