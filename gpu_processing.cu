#include "header.h"

#define TILE_SIZE 16
#define FILTER_SIZE 5
#define RADIUS (FILTER_SIZE / 2)
#define SHARED_SIZE (TILE_SIZE + FILTER_SIZE - 1)

__global__ void PictureDevice_FILTER(png_byte *d_In, png_byte *d_Out, int height, int width, float *d_filt) {
    // Calculate thread coordinates
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Col = blockIdx.x * TILE_SIZE + tx;
    int Row = blockIdx.y * TILE_SIZE + ty;

    // Define shared memory for the filter and a tile of the image
    __shared__ float shared_filt[FILTER_SIZE * FILTER_SIZE]; // 5x5 filter
    __shared__ png_byte shared_tile[SHARED_SIZE * SHARED_SIZE * 3]; // Tile of the image (3 color channels)

    // Load filter into shared memory
    if (tx < FILTER_SIZE && ty < FILTER_SIZE) {
        shared_filt[ty * FILTER_SIZE + tx] = d_filt[ty * FILTER_SIZE + tx];
    }

    // Load image tile into shared memory
    for (int color = 0; color < 3; color++) {
        int global_x = Col - RADIUS;
        int global_y = Row - RADIUS;
        int shared_x = tx;
        int shared_y = ty;

        // Handle the halo region
        for (int i = 0; i < SHARED_SIZE; i += TILE_SIZE) {
            for (int j = 0; j < SHARED_SIZE; j += TILE_SIZE) {

                // Check if the pixel is within the image bounds
                if ((shared_x + i) < SHARED_SIZE && (shared_y + j) < SHARED_SIZE) {

                    // Coordinate variables for the global image
                    int gx = global_x + i;
                    int gy = global_y + j;

                    // Check if the pixel is within the image bounds
                    if (gx >= 0 && gx < width && gy >= 0 && gy < height) {

                        // Copy the pixel to shared memory
                        shared_tile[((shared_y + j) * SHARED_SIZE + (shared_x + i)) * 3 + color] = d_In[(gy * width + gx) * 3 + color];
                    } else {
                        // Set the pixel to 0 if it is outside the image bounds
                        shared_tile[((shared_y + j) * SHARED_SIZE + (shared_x + i)) * 3 + color] = 0;
                    }
                }
            }
        }
    }

    // Ensure all threads in the block have loaded the data before proceeding
    __syncthreads();

    // Check if the thread is within the image bounds
    if (Row < height && Col < width) {
        float out;
        png_byte b;

        // Loop over the three color channels
        for (int color = 0; color < 3; color++) {
            out = 0.0;
            // Loop over the filter window centered on the current pixel
            for (int i = -RADIUS; i <= RADIUS; i++) {
                for (int j = -RADIUS; j <= RADIUS; j++) {
                    out += shared_filt[(i + RADIUS) * FILTER_SIZE + (j + RADIUS)] *
                           shared_tile[((ty + i + RADIUS) * SHARED_SIZE + (tx + j + RADIUS)) * 3 + color];
                }
            }
            // Clamp the result to the range [0, 255]
            b = (png_byte)fminf(fmaxf(out, 0.0), 255.0);
            d_Out[(Row * width + Col) * 3 + color] = b;
        }
    }
}

void execute_jobs_gpu(PROCESSING_JOB **jobs) {
    int count = 0;
    float *d_sharpen_filter = nullptr, *d_box_blur_filter = nullptr, *d_edge_filter = nullptr;
    png_byte *d_In = nullptr, *d_Out = nullptr;
    size_t maxNumPixels = 0;

    // Determine the maximum number of pixels to allocate memory once
    while (jobs[count] != nullptr) {
        size_t numPixels = jobs[count]->height * jobs[count]->width * 3; // 3 for RGB channels
        if (numPixels > maxNumPixels) {
            maxNumPixels = numPixels;
        }
        count++;
    }

    // Allocate memory on the device
    cudaMalloc((void **)&d_In, maxNumPixels * sizeof(png_byte));
    cudaMalloc((void **)&d_Out, maxNumPixels * sizeof(png_byte));
    cudaMalloc((void **)&d_sharpen_filter, 25 * sizeof(float));
    cudaMalloc((void **)&d_box_blur_filter, 25 * sizeof(float));
    cudaMalloc((void **)&d_edge_filter, 25 * sizeof(float));

    // Copy filters to device
    cudaMemcpy(d_sharpen_filter, sharpen_filter, 25 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_box_blur_filter, box_blur_filter, 25 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_filter, edge_detect_filter, 25 * sizeof(float), cudaMemcpyHostToDevice);

    // Initial creation of data and copying of data for the first image job
    char* filename = jobs[0]->source_name;
    size_t numPixels = jobs[0]->height * jobs[0]->width * 3; // 3 for RGB channels
    cudaMemcpy(d_In, jobs[0]->source_raw, numPixels * sizeof(png_byte), cudaMemcpyHostToDevice);
    dim3 blocks((jobs[0]->width - 4 + 15) / 16, (jobs[0]->height - 4 + 15) / 16);
    dim3 threads(16, 16);

    float *d_filter;
    switch (jobs[0]->processing_algo) {
        case SHARPEN:
            d_filter = d_sharpen_filter;
            break;
        case BLUR:
            d_filter = d_box_blur_filter;
            break;
        case EDGE:
            d_filter = d_edge_filter;
            break;
        default:
            return; // Invalid processing algorithm
    }

    // Run the kernel for the first image
    PictureDevice_FILTER<<<blocks, threads>>>(d_In, d_Out, jobs[0]->height, jobs[0]->width, d_filter);

    // Copy result back to host
    cudaMemcpy(jobs[0]->dest_raw, d_Out, numPixels * sizeof(png_byte), cudaMemcpyDeviceToHost);

    // Start loop at 1 since we already processed the first image
    count = 1;
    while (jobs[count] != nullptr) {

        char* current_filename = jobs[count]->source_name;

        // If the filename is different, we need to copy the new image to the device
        if (strcmp(filename, current_filename) != 0) {
            filename = current_filename;
            numPixels = jobs[count]->height * jobs[count]->width * 3; // 3 for RGB channels
            cudaMemcpy(d_In, jobs[count]->source_raw, numPixels * sizeof(png_byte), cudaMemcpyHostToDevice);
            blocks = dim3((jobs[count]->width - 4 + 15) / 16, (jobs[count]->height - 4 + 15) / 16);
        }

        // Determine filter
        switch (jobs[count]->processing_algo) {
            case SHARPEN:
                d_filter = d_sharpen_filter;
                break;
            case BLUR:
                d_filter = d_box_blur_filter;
                break;
            case EDGE:
                d_filter = d_edge_filter;
                break;
            default:
                return; // Invalid processing algorithm
        }

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
