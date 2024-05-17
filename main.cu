#include "header.h"

__global__ void dummyKernel() {
    // Empty kernel that does nothing
}

void activateGPU() {
    dummyKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}

long long timeInMilliseconds(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return (((long long)tv.tv_sec) * 1000) + (tv.tv_usec / 1000);
}


int main(int argc, char **argv)
{

    // Function that turns the GPU on to ensure there is no overhead when it turns on later when processing images
    activateGPU();

    long long processing_start = timeInMilliseconds();
    PROCESSING_JOB **jobs = prepare_jobs(argv[1]);
    long long processing_end = timeInMilliseconds();

    printf("Execution Time of loading part: %lld ms\n", processing_end - processing_start);

    long long gpu_start = timeInMilliseconds();
    execute_jobs_gpu(jobs);
    long long gpu_end = timeInMilliseconds();

    printf("Execution Time of processing part: %lld ms\n",gpu_end-gpu_start);

    long long cpu_start = timeInMilliseconds();
    execute_jobs_cpu(jobs);
    long long cpu_end = timeInMilliseconds();

    printf("Execution Time of cpu part: %lld ms\n",cpu_end-cpu_start);

    long long write_start = timeInMilliseconds();
    write_jobs_output_files(jobs);
    long long write_end = timeInMilliseconds();

    printf("Execution Time of writing part: %lld ms\n", write_end - write_start);
}
