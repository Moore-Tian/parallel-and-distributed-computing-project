import subprocess


def run(procs, length):
    command = f"mpiexec -n {procs} python psrs.py -l {length}"
    result = subprocess.run(command.split(), capture_output=True, text=True)
    t_p, correct_sort_str = result.stdout.strip().split()
    return float(t_p), correct_sort_str == 'True'


test_times = 10

length_list = [1000, 10000, 100000, 1000000]
proc_list = [1, 2, 4, 8, 16]

serial_time = 0
for length in length_list:
    for procs in proc_list:
        results = [run(procs, length) for _ in range(test_times)]
        time, Correct = zip(*results)
        time = sum(time) / test_times
        if procs == 1:
            serial_time = time
        Status = "correct" if all(Correct) else "incorrect"
        print(procs, length, ":", time, serial_time / time, Status)
