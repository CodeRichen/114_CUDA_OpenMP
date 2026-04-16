void gemm_cuda_core_run(int N);
void gemm_tensor_core_run(int N);

int main()
{
    int N = 256;   // 你可以改成 512 或 1024

    gemm_cuda_core_run(N);
    gemm_tensor_core_run(N);

    return 0;
}
