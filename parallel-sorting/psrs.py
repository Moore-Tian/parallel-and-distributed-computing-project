from mpi4py import MPI
import random
import bisect
import argparse
from itertools import chain


class PSRS():
    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-l', '--length', type=int, required=True, help='Length of the array')
        return parser.parse_args()

    def data_distribution(self, data, num_proc):
        partition_size = len(data) // num_proc
        partitions = [data[i: i + partition_size] for i in range(0, (num_proc - 1) * partition_size, partition_size)]
        partitions.append(data[(num_proc - 1) * partition_size:])
        return partitions

    def select_sampling_points(self, data, num_samples):
        return [data[i] for i in range(0, len(data), max(len(data) // num_samples, 1))][:num_samples]

    def select_pivots(self, samples, num_pivots):
        samples_flattened = sorted(chain(*samples))
        return samples_flattened[num_pivots:len(samples_flattened):num_pivots][:num_pivots - 1]

    def partition_local_data(self, local_data, pivot_values):
        partitions = []
        start_index = 0
        for pivot in pivot_values:
            end_index = bisect.bisect_left(local_data, pivot, start_index)
            partitions.append(local_data[start_index:end_index])
            start_index = end_index
        partitions.append(local_data[start_index:])
        return partitions

    def sort_and_merge(self, data_segments):
        return sorted(chain(*data_segments))

    def run(self):
        mpi_comm = MPI.COMM_WORLD
        num_proc = mpi_comm.Get_size()
        rank = mpi_comm.Get_rank()

        # 如果为主进程，则进行数据的生成和分配
        if rank == 0:
            random.seed(666)
            args = self.parse_arguments()
            unsorted_array = [random.randint(0, 10000000) for _ in range(args.length)]
            # 生成标答，用于最后验证结果
            sorted_array = sorted(unsorted_array)
            start_time = MPI.Wtime()
            data_partitions = self.data_distribution(unsorted_array, num_proc)
        else:
            data_partitions = None

        # 数据被分配给各个进程，进行排序、采样和合并
        local_data = mpi_comm.scatter(data_partitions, root=0)
        local_sorted_data = self.sort_and_merge([local_data])
        sample_points = self.select_sampling_points(local_sorted_data, num_proc)
        gathered_samples = mpi_comm.gather(sample_points, root=0)

        # 在主进程上选择主元
        if rank == 0:
            pivot_values = self.select_pivots(gathered_samples, num_proc)
        else:
            pivot_values = None

        # 将主元广播给所有进程，各进程根据主元划分本地数据，之后合并排序获得最终数据
        pivot_values = mpi_comm.bcast(pivot_values, root=0)
        local_partitions = self.partition_local_data(local_sorted_data, pivot_values)
        received_partitions = mpi_comm.alltoall(local_partitions)
        final_sorted_data = self.sort_and_merge(received_partitions)
        final_sorted_data = mpi_comm.gather(final_sorted_data, root=0)

        if rank == 0:
            merged_final_data = self.sort_and_merge(final_sorted_data)
            cost_time = MPI.Wtime() - start_time
            if_correct = (merged_final_data == sorted_array)
            print(cost_time, if_correct)


psrs = PSRS()
psrs.run()
