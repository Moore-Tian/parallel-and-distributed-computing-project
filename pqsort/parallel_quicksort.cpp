#include<stdio.h>
#include<omp.h>
#include <iostream>
#include<time.h>
#include<stdlib.h>
using namespace std;


int Partition(int* arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++)
        if (arr[j] < pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}
 
void quickSort(int* data, int start, int end)  //并行快排
{
    if (start < end) {
        int pos = Partition(data, start, end);
        #pragma omp parallel sections    //设置并行区域
        {
            #pragma omp section          //该区域对前部分数据进行排序
            quickSort(data, start, pos - 1);
            #pragma omp section          //该区域对后部分数据进行排序
            quickSort(data, pos + 1, end);
        }
    }
}
 
int main(int argc, char* argv[])
{
    int n = atoi(argv[2]), i;   //线程数
    int size = atoi(argv[1]);   //数据大小
    int* num = (int*)malloc(sizeof(int) * size);
 
    srand(time(NULL) + rand());   //生成随机数组
    for (i = 0; i < size; i++)
        num[i] = rand();
    omp_set_num_threads(n);   //设置线程数
    double starttime = omp_get_wtime();
    quickSort(num, 0, size - 1);   //并行快排
    double endtime = omp_get_wtime();
 
    printf(" %d %d %lf\n", size, n, endtime - starttime);
    return 0;
}