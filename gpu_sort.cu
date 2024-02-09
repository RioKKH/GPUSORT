#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

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


int main()
{
    const int arraySize = 512;
    int hostData[arraySize];
    int hostData_cpu[arraySize];
    int hostData_gpu[arraySize];
    int *deviceData;
    float bubbleElapsedTimeCPU = 0.0f;
    float bubbleElapsedTimeGPU = 0.0f;
    float quickElapsedTimeCPU = 0.0f;
    float quickElapsedTimeGPU = 0.0f;

    // イベントのクリエイト
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 乱数のシートを設定
    srand(time(NULL));

    // 配列を初期化
    for (int i = 0; i < arraySize; i++)
    {
        hostData[i] = rand() % arraySize; // 0 から arraySize-1 の乱数を生成
    }

    // 初期状態を表示
    printf("Initial:");
    printResult(hostData, 10);

    // CPUでバブルソートを実行 -----------------------------------------------
    copy_array(hostData, hostData_cpu, arraySize);

    cudaEventRecord(start, 0);
    bubbleSortCPU(hostData_cpu, arraySize);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&bubbleElapsedTimeCPU, start, stop);

    printf("bubble_sort_cpu: ");
    printResult(hostData_cpu, 10);
    // ---------------------------------------------------------------------------

    // CPUでクイックソートを実行 -----------------------------------------------
    copy_array(hostData, hostData_cpu, arraySize);

    cudaEventRecord(start, 0);
    quickSortCPU(hostData_cpu, 0, arraySize - 1);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&quickElapsedTimeCPU, start, stop);

    printf("quick_sort_cpu: ");
    printResult(hostData_cpu, 10);
    // ---------------------------------------------------------------------------

    // GPUでバブルソートを実行 -------------------------------------------------
    cudaMalloc((void **)&deviceData, arraySize * sizeof(int));
    cudaMemcpy(deviceData, hostData, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);

    bubbleSort<<<1, 1>>>(deviceData, arraySize);
    cudaMemcpy(hostData_gpu, deviceData, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&bubbleElapsedTimeGPU, start, stop);

    printf("bubble_sort_gpu: ");
    printResult(hostData_gpu, 10);
    // ---------------------------------------------------------------------------

    // GPUでクイックソートを実行 -------------------------------------------------
    cudaMalloc((void **)&deviceData, arraySize * sizeof(int));
    cudaMemcpy(deviceData, hostData, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);

    quickSortKernel<<<1, 1>>>(deviceData, 0, arraySize - 1);
    cudaMemcpy(hostData_gpu, deviceData, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&quickElapsedTimeGPU, start, stop);

    printf("quick_sort_gpu: ");
    printResult(hostData_gpu, 10);
    // ---------------------------------------------------------------------------

    printf("Elapsed time: %f,%f,%f,%f\n", 
            bubbleElapsedTimeCPU, quickElapsedTimeCPU,
            bubbleElapsedTimeGPU, quickElapsedTimeGPU);

    cudaFree(deviceData);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
