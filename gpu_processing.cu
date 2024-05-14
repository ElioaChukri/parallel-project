#include "header.h"

__global__ void PictureDevice_FILTER(png_byte *d_In, png_byte *d_Out, int height, int width, float *d_filt) {
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Row >= 2 && Row < height - 2 && Col >= 2 && Col < width - 2) {
        float out;
        png_byte b;

        for (int color = 0; color < 3; color++) {
            out = 0.0;
            for (int i = -2; i <= 2; i++) {
                for (int j = -2; j <= 2; j++) {
                    out += d_filt[(i+2) * 5 + (j+2)] * d_In[((Row + i) * width + (Col + j)) * 3 + color];
                }
            }
            b = (png_byte)fminf(fmaxf(out, 0.0), 255.0);
            d_Out[(Row * width + Col) * 3 + color] = b;
        }
    }
}

void execute_jobs_gpu(PROCESSING_JOB **jobs) {
    int count = 0;
    float *d_filter = nullptr;
    png_byte *d_In = nullptr, *d_Out = nullptr;
    size_t maxNumPixels = 0;

    // Determine the maximum number of pixels to allocate memory once
    while (jobs[count] != NULL) {
        size_t numPixels = jobs[count]->height * jobs[count]->width * 3; // 3 for RGB channels
        if (numPixels > maxNumPixels) {
            maxNumPixels = numPixels;
        }
        count++;
    }

    // Allocate memory on the device
    cudaMalloc((void **)&d_In, maxNumPixels * sizeof(png_byte));
    cudaMalloc((void **)&d_Out, maxNumPixels * sizeof(png_byte));
    cudaMalloc((void **)&d_filter, 25 * sizeof(float)); // Assuming a 5x5 filter

    count = 0;
    while (jobs[count] != NULL) {
        size_t numPixels = jobs[count]->height * jobs[count]->width * 3; // 3 for RGB channels
        float *h_filter = getAlgoFilterByType(jobs[count]->processing_algo);

        // Copy data from host to device
        cudaMemcpy(d_In, jobs[count]->source_raw, numPixels * sizeof(png_byte), cudaMemcpyHostToDevice);
        cudaMemcpy(d_filter, h_filter, 25 * sizeof(float), cudaMemcpyHostToDevice);

        // Set up the execution configuration
        dim3 blocks((jobs[count]->width - 4 + 15) / 16, (jobs[count]->height - 4 + 15) / 16);
        dim3 threads(16, 16);

        // Launch the kernel
        PictureDevice_FILTER<<<blocks, threads>>>(d_In, d_Out, jobs[count]->height, jobs[count]->width, d_filter);

        // Copy result back to host
        cudaMemcpy(jobs[count]->dest_raw, d_Out, numPixels * sizeof(png_byte), cudaMemcpyDeviceToHost);

        count++;
    }

    // Free device memory
    cudaFree(d_In);
    cudaFree(d_Out);
    cudaFree(d_filter);
}
