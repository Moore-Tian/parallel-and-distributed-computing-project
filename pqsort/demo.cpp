#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <omp.h>
#include <algorithm>

using namespace std;

// 分区函数
int partition(vector<int>& arr, int left, int right) {
    int pivot = arr[(left + right) / 2]; // 选择中间元素作为基准
    int i = left, j = right;
    while (i <= j) {
        while (arr[i] < pivot) i++;
        while (arr[j] > pivot) j--;
        if (i <= j) {
            swap(arr[i], arr[j]);
            i++;
            j--;
        }
    }
    return i;
}

// 快速排序函数
void quicksort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int pivot = partition(arr, left, right);
        #pragma omp parallel sections
        {
            #pragma omp section
            quicksort(arr, left, pivot - 1);
            #pragma omp section
            quicksort(arr, pivot, right);
        }
    }
}

int main() {
    vector<int> data_sizes = {1000, 5000, 10000, 100000};
    int num_threads[] = {1, 2, 4, 8};
    int num_runs = 5; // 每种情况下运行的次数

    for (int data_size : data_sizes) {
        vector<int> arr(data_size);
        srand(0); // 使用相同的随机种子,确保结果可重复
        for (int& x : arr) x = rand() % 1000000; // 生成0-999999之间的随机整数

        cout << "Data size: " << data_size << endl;

        double serial_time = 0;

        for (int threads : num_threads) {
            omp_set_num_threads(threads);
            vector<double> durations(num_runs);
            for (int i = 0; i < num_runs; i++) {
                auto start = chrono::high_resolution_clock::now();
                quicksort(arr, 0, data_size - 1);
                auto end = chrono::high_resolution_clock::now();
                durations[i] = chrono::duration_cast<chrono::milliseconds>(end - start).count();
            }

            // 计算平均运行时间
            double avg_duration = 0;
            for (double d : durations) avg_duration += d;
            avg_duration /= num_runs;
            cout << "Threads: " << threads << ", Time: " << avg_duration << " ms" << endl;

            // 计算加速比
            if (threads == 1) {
                serial_time = avg_duration;
            }
            cout << "Speedup: " << serial_time / avg_duration << endl;

            // 验证排序结果是否正确
            vector<int> sorted_arr = arr;
            sort(sorted_arr.begin(), sorted_arr.end());
            if (arr == sorted_arr) {
                cout << "Sorting is correct." << endl;
            } else {
                cout << "Sorting is incorrect." << endl;
            }
            cout << endl;
        }
    }

    return 0;
}