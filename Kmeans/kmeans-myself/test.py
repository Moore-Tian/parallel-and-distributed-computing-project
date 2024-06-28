import subprocess
import time 


def run(procs):
    command = f"a.exe {procs} Image_data/color1000000.txt"
    start_time = time.time()
    result = subprocess.run(command.split(), capture_output=True, text=True)
    end_time = time.time()
    return end_time - start_time


test_times = 10
proc_list = [1, 2, 4, 8, 16]

serial_time = 0

for procs in proc_list:
    times = [run(procs) for _ in range(test_times)]
    avg_time = sum(times) / test_times
    if procs == 1:
        serial_time = avg_time
    print(procs, ":", avg_time, serial_time / avg_time)
