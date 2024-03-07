#include <iostream>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

int main()
{
    const int size = 10; // 配列のサイズ
    const int N = 5;     // 抽出する要素数

    // ホスト側の配列
    // このコードは配列Ｂをソートキーとして昇順にならびかえ、
    // 対応する配列Ａの値も同じ順にならびかえる
    int h_A[size] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int h_B[size] = {900, 28, 17, 56, 45, 444, 23, 2, 125, 200}; // ソート基準の配列

    // デバイス側の配列
    int *d_A, *d_B, *d_sorted_A, *d_sorted_B;
    cudaMalloc(&d_A, sizeof(int) * size);
    cudaMalloc(&d_B, sizeof(int) * size);
    cudaMalloc(&d_sorted_A, sizeof(int) * size);
    cudaMalloc(&d_sorted_B, sizeof(int) * size);

    // デバイスへデータをコピー
    cudaMemcpy(d_A, h_A, sizeof(int) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(int) * size, cudaMemcpyHostToDevice);

    // CUBの一時ストレージ
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_B, d_sorted_B, d_A, d_sorted_A, size);

    // 一時ストレージ用のメモリを確保
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // SortPairs()はキーと値のペアをキーの順にソートする
    // 配列Ｂの値をキーとして、配列Ａの値をソートする
    // この場合、配列Ｂの値が小さい順に配列Ａの値がソートされる
    cub::DeviceRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_B, d_sorted_B, d_A, d_sorted_A, size);

    // ソートされたインデックスをホストにコピーする
    int h_original_A[size];
    int h_sorted_A[size];
    cudaMemcpy(h_original_A, d_A,      sizeof(int) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sorted_A, d_sorted_A, sizeof(int) * size, cudaMemcpyDeviceToHost);

    // 上位N要素のインデックスを表示する
    std::cout << "Top " << N << " indices:" << std::endl;
    /*
    for (const auto &i : h_sorted_A)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    */

    /*
    std::cout 
        << h_sorted_A[9] << h_sorted_A[8] 
        << h_sorted_A[7] << h_sorted_A[6]
        << h_sorted_A[5] << std::endl;
    */

    std::cout << "h_original_A: ";
    for (int i = 9; i >= 0; i--)
    {
        std::cout << h_original_A[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 9; i >= 0; i--)
    {
        std::cout << h_B[h_original_A[i]] << " ";
    }
    std::cout << std::endl;

    std::cout << "h_sorted_A: ";
    for (int i = 9; i >= 0; i--)
    // for (int i = 0; i < N; i++)
    {
        // std::cout << i << " ";
        std::cout << h_sorted_A[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 9; i >= 0; i--)
    {
        std::cout << h_B[h_sorted_A[i]] << " ";
    }
    std::cout << std::endl;

    // メモリ解放
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_sorted_A);
    cudaFree(d_sorted_B);
    cudaFree(d_temp_storage);

    return 0;
}

