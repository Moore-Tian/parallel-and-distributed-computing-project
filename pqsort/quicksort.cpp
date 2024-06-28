#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <climits>

using namespace std;

int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high]; // ѡ�����ұߵ�Ԫ����Ϊ��׼
    int i = low - 1; // ��ʼ��һ��ָ�룬ָ��Ȼ�׼С��Ԫ�ص�����λ��

    for (int j = low; j < high; j++)
        if (arr[j] < pivot) {
            i++;
            swap(arr[i], arr[j]); // �����ǰԪ��С�ڻ�׼�����������
        }
    swap(arr[i + 1], arr[high]); // ����׼Ԫ�طŵ��м�λ��
    return i + 1; // ���ػ�׼Ԫ�ص�λ��
}


// ����������
void quickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1); // �Ի�׼����Ԫ�ؽ��еݹ�����
        quickSort(arr, pi + 1, high); // �Ի�׼�Ҳ��Ԫ�ؽ��еݹ�����
    }
}

// ��ʼ�����ݣ���������
void initializeData(vector<int>& data) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, 1000000);
    generate(data.begin(), data.end(), [&]() { return dis(gen); });
}

// ִ�д��п������򣬲���ʱ
void serial_quickSort(vector<int>& data) {
    quickSort(data, 0, data.size() - 1);  
}

// ִ�в��п�������
void parallel_quickSort(vector<int>& data, int low, int high) {
    if (low < high) {
        int pivotpos = partition(data, low, high);   
#pragma omp parallel sections
        {
#pragma omp section
            quickSort(data, low, pivotpos - 1);
#pragma omp section
            quickSort(data, pivotpos + 1, high);
        }
    }  
}

// ��֤����Ľ���Ƿ��봮����������ͬ
void verifyResults(const vector<int>& sorted_data, const vector<int>& reference_data, size_t size) {
    for (int i = 0; i < size; ++i) {
        if (sorted_data[i] != reference_data[i]) {
            cout << "Mismatch at " << i << endl;
            return;
        }
    }
    cout << "OK!!\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <size_of_data>\n";
        return 1;
    }

    int size = atoi(argv[1]);
    vector<int> data(size), reference_data(size);

    initializeData(data);
    copy(data.begin(), data.end(), reference_data.begin());

    auto start_serial = chrono::high_resolution_clock::now();
    serial_quickSort(data);
    auto end_serial = chrono::high_resolution_clock::now();
    auto time_serial = 0.0;

    int threads[] = { 1, 2, 4, 8, 16};
    for (int numThr : threads) {
        vector<int> temp_data(size);
        copy(reference_data.begin(), reference_data.end(), temp_data.begin());

        omp_set_num_threads(numThr);
        auto start_parallel = chrono::high_resolution_clock::now();
        parallel_quickSort(temp_data, 0, temp_data.size() - 1);
        auto end_parallel = chrono::high_resolution_clock::now();
        auto time_parallel = chrono::duration_cast<chrono::nanoseconds>(end_parallel - start_parallel).count() / 1e9;

        if(numThr == 1) time_serial = time_parallel;

        cout << "Parallel computation for " << size << " elements with " << numThr << " threads: " << time_parallel << "s\n";
        double speed_up = time_serial / time_parallel;
        cout << "Speed up: " << speed_up << '\n';

        verifyResults(temp_data, data, data.size());
    }
    return 0;
}
