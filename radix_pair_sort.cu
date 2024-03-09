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
    int h_B[size] = {100, 90, 80, 70, 60, 50, 40, 30, 20, 10}; // ソート基準の配列
    // ホスト側の配列
    int *h_sorted_A = (int *)malloc(sizeof(int) * size);
    int *h_sorted_B = (int *)malloc(sizeof(int) * size);
    // ホスト側の戻り値を格納する配列
    int *h_returned_A = (int *)malloc(sizeof(int) * size);
    int *h_returned_B = (int *)malloc(sizeof(int) * size);

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

    // ソートに使った配列をホスト側にコピー
    cudaMemcpy(h_returned_A, d_A,      sizeof(int) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_returned_B, d_B,      sizeof(int) * size, cudaMemcpyDeviceToHost);

    // ソートされた配列をホスト側にコピー
    cudaMemcpy(h_sorted_A, d_sorted_A, sizeof(int) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sorted_B, d_sorted_B, sizeof(int) * size, cudaMemcpyDeviceToHost);

    // 結果の表示
    std::cout << "h_returned:" << std::endl;
    for (int i = 0; i <= 9; i++) { std::cout << h_returned_A[i] << " "; }
    std::cout << std::endl;
    for (int i = 0; i <= 9; i++) { std::cout << h_returned_B[i] << " "; }
    std::cout << std::endl;

    std::cout << "h_sorted: " << std::endl;
    for (int i = 0; i <= 9; i++) { std::cout << h_sorted_A[i] << " "; }
    std::cout << std::endl;
    for (int i = 0; i <= 9; i++) { std::cout << h_sorted_B[i] << " "; }
    std::cout << std::endl;

    // メモリ解放
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_sorted_A);
    cudaFree(d_sorted_B);
    cudaFree(d_temp_storage);

    free(h_sorted_A);
    free(h_sorted_B);
    free(h_returned_A);
    free(h_returned_B);


    return 0;
}

