#include <iostream>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

/**
データをソートするカーネル
これはコンパイルエラーになる。なぜならば、cub::DeviceRadixSort::SortKeys()は
ホストコードであり、デバイスコードではないからである。
__global__ void sortKernel(
        int *d_input, int *d_output, size_t numItems, 
        void *d_tempStorage, size_t tempStorageBytes)
{
    cub::DeviceRadixSort::SortKeys(d_tempStorage, tempStorageBytes, d_input, d_output, numItems);
}
*/

int main()
{
    int numItems = 10;
    int h_input[] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    int *d_input, *d_output;

    // デバイスメモリを確保
    cudaMalloc(&d_input, sizeof(int) * numItems);
    cudaMalloc(&d_output, sizeof(int) * numItems);

    // ホストからデバイスへデータをコピーする
    cudaMemcpy(d_input, h_input, sizeof(int) * numItems, cudaMemcpyHostToDevice);

    // 一時ストレージ用のメモリサイズを取得する
    // nullptrとNULLの違いは、nullptrはC++11から導入されたポインタ型のnullリテラルで、
    // NULLはC++03までのC++で使われていたポインタ型のnullリテラルです。
    // どちらもポインタ型のnullリテラルですが、nullptrはポインタ型のnullリテラルとして
    // 専用に用意されたものです。
    // 一方、NULLはC言語から継承されたもので、整数型の0をポインタ型にキャストしたものです。
    // そのため、nullptrはポインタ型のnullリテラルとして使われることが保証されていますが、
    // NULLは整数型の0をポインタ型にキャストしたものであるため、ポインタ型のnullリテラル
    // として使われることが保証されていません。
    void *d_tempStorage = nullptr;
    size_t tempStorageBytes = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, tempStorageBytes, d_input, d_output, numItems);

    // 一時ストレージ用のメモリを確保
    cudaMalloc(&d_tempStorage, tempStorageBytes);

    // ソートを実行する
    // sortKernel<<<1, 1>>>(d_input, d_output, numItems, d_tempStorage, tempStorageBytes);
    // 以下のようにホストコードで実行する
    cub::DeviceRadixSort::SortKeys(d_tempStorage, tempStorageBytes, d_input, d_output, numItems);

    // ソートされたデータをホストにコピーする
    int h_output[numItems];
    cudaMemcpy(h_output, d_output, sizeof(int) * numItems, cudaMemcpyDeviceToHost);

    // 結果を表示する
    for (int i = 0; i < numItems; i++)
    {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // メモリを解放する
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_tempStorage);

    return 0;
}

