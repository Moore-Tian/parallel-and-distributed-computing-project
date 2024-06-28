import subprocess


def run(procs, length):
    command = f"a.exe {length} {procs}"
    result = subprocess.run(command.split(), capture_output=True, text=True)
    data_size, proc_num, time = result.stdout.strip().split()
    return float(time)


test_times = 10

length_list = [1000, 10000, 100000, 1000000]
proc_list = [1, 2, 4, 8, 16]

serial_time = 0
for length in length_list:
    for procs in proc_list:
        times = [run(procs, length) for _ in range(test_times)]
        time = sum(times) / test_times
        if procs == 1:
            serial_time = time
        print(procs, length, ":", time, serial_time / time)
