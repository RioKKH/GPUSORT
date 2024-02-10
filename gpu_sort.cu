#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

void bubbleSortCPU(int *data, int count)
{
    int temp;
    for (int i = 0; i < count - 1; i++)
    {
        for (int j = 0; j < count - i - 1; j++)
        {
            if (data[j] > data[j + 1])
            {
                temp = data[j];
                data[j] = data[j + 1];
                data[j + 1] = temp;
            }
        }
    }
}

void quickSortCPU(int *data, int left, int right)
{
    int i = left, j = right;
    int pivot = data[(left + right) / 2];

    while (i <= j)
    {
        while (data[i] < pivot) i++;
        while (data[j] > pivot) j--;
        if (i <= j)
        {
            int tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
            i++;
            j--;
        }
    }
    if (left < j) quickSortCPU(data, left, j);
    if (i < right) quickSortCPU(data, i, right);

    /**
    if (left < right)
    {
        int pivot = data[right];
        int i = left -1;
        for (int j = left; j < right; j++)
        {
            if (data[j] < pivot)
            {
                i++;
                int tmp = data[i];
                data[i] = data[j];
                data[j] = tmp;
            }
        }
        int tmp = data[i + 1];
        data[i + 1] = data[right];
        data[right] = tmp;
        quickSortCPU(data, left, i);
        quickSortCPU(data, i + 1, right - 1);
    }
    */
}

/**
  * バブルソートを実行するカーネル
  * 計算量は O(n^2) なので、大きなデータに対しては非効率
  * @param data ソートするデータ
  * @param count データの個数
  */
__global__ void bubbleSort(int *data, int count)
{
    int temp;
    for (int i = 0; i < count - 1; i++)
    {
        for (int j = 0; j < count - i - 1; j++)
        {
            if (data[j] > data[j + 1])
            {
                temp = data[j];
                data[j] = data[j + 1];
                data[j + 1] = temp;
            }
        }
    }
}

/**
  * クイックソートを実行するカーネル
  * 計算量は O(n log n) なので、大きなデータに対しても効率的
  * @param data ソートするデータ
  * @param count データの個数
  */
__device__ void quickSort(int *data, int left, int right)
{
    int i = left, j = right;
    int tmp;
    int pivot = data[(left + right) / 2];

    /* パーティション分割 */
    while (i <= j)
    {
        while (data[i] < pivot)
            i++;
        while (data[j] > pivot)
            j--;
        if (i <= j)
        {
            tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
            i++;
            j--;
        }
    }

    /* 再帰的にソート */
    if (left < j)
        quickSort(data, left, j);
    if (i < right)
        quickSort(data, i, right);
}

__global__ void quickSortKernel(int *data, int left, int right)
{
    quickSort(data, left, right);
}


__global__ void radixSort(int *data, int count)
{
}


// 最初のcount要素と最後のcount要素を表示
void printResult(int *data, int count)
{
    printf("Result:");
    // 最初のcount要素だけ表示
    for (int i = 0; i < count; i++)
    {
        printf("%d ", data[i]);
    }
    // 最後のcount要素だけ表示
    printf("...");
    for (int i = count; i > 0; i--)
    {
        printf("%d ", data[512 - i]);
    }
    printf("\n");
}

int copy_array(int *src, int *dst, int count)
{
    for (int i = 0; i < count; i++)
    {
        dst[i] = src[i];
    }
    return 0;
}


int main(int argc, char **argv)
{
    const int arraySize = 512;
    int hostData[arraySize];
    int hostData_cpu[arraySize];
    int hostData_gpu[arraySize];
    int *deviceInput, *deviceOutput;

    float bubbleElapsedTimeCPU = 0.0f;
    float bubbleElapsedTimeGPU = 0.0f;
    float quickElapsedTimeCPU = 0.0f;
    float quickElapsedTimeGPU = 0.0f;
    float radixElapsedTimeGPUThrust = 0.0f;
    float radixElapsedTimeGPUCUB = 0.0f;

    thrust::host_vector<int> h_vec(arraySize);
    thrust::device_vector<int> d_vec(arraySize);


    // これは、実行時に指定された回数だけ計測を繰り返すためのもの
    // 通常は 1 回で十分
    int measureCount = argv[1] ? atoi(argv[1]) : 1;

    // イベントのクリエイト
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int k = 0; k < measureCount; k++)
    {

        // 乱数のシートを設定
        srand(time(NULL));

        // 配列を初期化
        for (int i = 0; i < arraySize; i++)
        {
            hostData[i] = rand() % arraySize; // 0 から arraySize-1 の乱数を生成
            h_vec[i] = hostData[i];
        }

        // 初期状態を表示
#ifdef DEBUG
        printf("Initial:");
        printResult(hostData, 10);
#endif

        // CPUでバブルソートを実行 -----------------------------------------------
        copy_array(hostData, hostData_cpu, arraySize);

        cudaEventRecord(start, 0);
        bubbleSortCPU(hostData_cpu, arraySize);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&bubbleElapsedTimeCPU, start, stop);

#ifdef DEBUG
        printf("bubble_sort_cpu: ");
        printResult(hostData_cpu, 10);
#endif
        // ---------------------------------------------------------------------------

        // CPUでクイックソートを実行 -----------------------------------------------
        copy_array(hostData, hostData_cpu, arraySize);

        cudaEventRecord(start, 0);
        quickSortCPU(hostData_cpu, 0, arraySize - 1);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&quickElapsedTimeCPU, start, stop);

#ifdef DEBUG
        printf("quick_sort_cpu: ");
        printResult(hostData_cpu, 10);
#endif
        // ---------------------------------------------------------------------------

        // GPUでバブルソートを実行 -------------------------------------------------
        cudaMalloc((void **)&deviceInput, arraySize * sizeof(int));
        cudaMemcpy(deviceInput, hostData, arraySize * sizeof(int), cudaMemcpyHostToDevice);

        cudaEventRecord(start, 0);

        bubbleSort<<<1, 1>>>(deviceInput, arraySize);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&bubbleElapsedTimeGPU, start, stop);

        cudaMemcpy(hostData_gpu, deviceInput, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

#ifdef DEBUG
        printf("bubble_sort_gpu: ");
        printResult(hostData_gpu, 10);
#endif
        // ---------------------------------------------------------------------------

        // GPUでクイックソートを実行 -------------------------------------------------
        cudaMalloc((void **)&deviceInput, arraySize * sizeof(int));
        cudaMemcpy(deviceInput, hostData, arraySize * sizeof(int), cudaMemcpyHostToDevice);

        cudaEventRecord(start, 0);

        quickSortKernel<<<1, 1>>>(deviceInput, 0, arraySize - 1);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&quickElapsedTimeGPU, start, stop);

        cudaMemcpy(hostData_gpu, deviceInput, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

#ifdef DEBUG
        printf("quick_sort_gpu: ");
        printResult(hostData_gpu, 10);
#endif
        // ---------------------------------------------------------------------------


        // GPUでRADIX sortを実行(thrust) -----------------------------------------
        d_vec = h_vec;

        cudaEventRecord(start, 0);
        thrust::sort(d_vec.begin(), d_vec.end());
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&radixElapsedTimeGPUThrust, start, stop);

        thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
#ifdef DEBUG
        printf("RADIX_sort_gpu: ");
        printResult(&h_vec[0], 10);
#endif
        // ---------------------------------------------------------------------------

        // GPUでRADIX sortを実行(cub) -----------------------------------------

        // CUBのための一時領域を確保
        void *deviceTempStorage = nullptr;
        size_t tempStorageBytes = 0;
        cub::DeviceRadixSort::SortKeys(nullptr, tempStorageBytes, deviceInput, deviceOutput, arraySize);
        cudaMalloc(&deviceInput, arraySize * sizeof(int));
        cudaMalloc(&deviceOutput, arraySize * sizeof(int));
        // 一時ストレージ用のメモリを確保
        cudaMalloc(&deviceTempStorage, tempStorageBytes);

        cudaEventRecord(start, 0);
        cub::DeviceRadixSort::SortKeys(deviceTempStorage, tempStorageBytes, deviceInput, deviceOutput, arraySize);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&radixElapsedTimeGPUCUB, start, stop);

        cudaMemcpy(hostData_gpu, deviceOutput, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

#ifdef DEBUG
        printf("RADIX_sort_gpu: ");
        printResult(hostData_gpu, 10);
#endif
        // ---------------------------------------------------------------------------

        printf("%f,%f,%f,%f,%f,%f\n", 
                bubbleElapsedTimeCPU, quickElapsedTimeCPU,
                bubbleElapsedTimeGPU, quickElapsedTimeGPU,
                radixElapsedTimeGPUThrust, radixElapsedTimeGPUCUB);
    }

    cudaFree(deviceInput);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
