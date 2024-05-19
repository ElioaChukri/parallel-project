#include "header.h"

// struct to handle writing images in different threads
typedef struct {
    PROCESSING_JOB **jobs;
    int *job_index;
    int total_jobs;
    pthread_mutex_t *mutex;
} WriteThreadData;

int count_lines(char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening job file <%s>.\n", filename);
        exit(-1);
    }
    int lineCount = 0;
    int ch;

    while ((ch = fgetc(file)) != EOF) {
        if (ch == '\n') {
            lineCount++;
        }
    }

    fclose(file);
    return lineCount;
}

typedef struct {
    char *filename;
    PROCESSING_JOB **processing_job;
    int start_idx;
    int end_idx;
} ThreadData;

void* prepare_job(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    char input_filename[100], processing_algo[100], output_filename[100];
    FILE *file = fopen(data->filename, "r");
    if (file == NULL) {
        printf("Error opening job file <%s>.\n", data->filename);
        exit(-1);
    }

    // Skip to the start index
    for (int i = 0; i < data->start_idx; i++) {
        fscanf(file, "%s %s %s", input_filename, processing_algo, output_filename);
    }

    for (int i = data->start_idx; i < data->end_idx; i++) {
        fscanf(file, "%s %s %s", input_filename, processing_algo, output_filename);

        PNG_RAW *png_raw = read_png(input_filename);
        if (png_raw->pixel_size != 3) {
            printf("Error, png file <%s> must be on 3 Bytes per pixel (not %d)\n", input_filename, png_raw->pixel_size);
            exit(0);
        }

        data->processing_job[i] = (PROCESSING_JOB *)malloc(sizeof(PROCESSING_JOB));
        data->processing_job[i]->source_name = strdup(input_filename);
        data->processing_job[i]->dest_name = strdup(output_filename);
        data->processing_job[i]->width = png_raw->width;
        data->processing_job[i]->height = png_raw->height;
        data->processing_job[i]->info_ptr = png_raw->info_ptr;
        data->processing_job[i]->pixel_size = png_raw->pixel_size;
        data->processing_job[i]->source_raw = png_raw->buf;
        data->processing_job[i]->dest_raw = (png_byte *)malloc(png_raw->width * png_raw->height * png_raw->pixel_size * sizeof(png_byte));

        clear_buf(data->processing_job[i]->dest_raw, png_raw->height, png_raw->width);
        data->processing_job[i]->processing_algo = getAlgoByName(processing_algo);
    }

    fclose(file);
    pthread_exit(NULL);
}

PROCESSING_JOB** prepare_jobs(char *filename) {
    int nb_jobs = count_lines(filename);

    // Allocate memory for the jobs
    PROCESSING_JOB **processing_job = (PROCESSING_JOB **)malloc((nb_jobs + 1) * sizeof(PROCESSING_JOB*));

    int num_threads = 9; // Adjust based on your CPU cores. 9 is the optimal number for the given jobs.txt since there are 9 tasks
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    // Variables to handle the distribution of jobs among threads
    int jobs_per_thread = nb_jobs / num_threads;
    int remaining_jobs = nb_jobs % num_threads;
    int current_start_idx = 0;

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].filename = filename;
        thread_data[i].processing_job = processing_job;
        thread_data[i].start_idx = current_start_idx;
        current_start_idx += jobs_per_thread + (i < remaining_jobs ? 1 : 0);
        thread_data[i].end_idx = current_start_idx;

        pthread_create(&threads[i], NULL, prepare_job, (void *)&thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    processing_job[nb_jobs] = NULL;
    return processing_job;
}

void* write_job(void *arg) {
    WriteThreadData *data = (WriteThreadData *)arg;
    int job_idx;

    // Allocate PNG_RAW once per thread to avoid repeated allocations and deallocations
    PNG_RAW *png_raw = (PNG_RAW *)malloc(sizeof(PNG_RAW));

    while (1) {
        // Fetch the next job index
        pthread_mutex_lock(data->mutex);
        job_idx = *(data->job_index);
        if (job_idx >= data->total_jobs) {
            pthread_mutex_unlock(data->mutex);
            break;
        }
        (*(data->job_index))++;
        pthread_mutex_unlock(data->mutex);

        // Process the job
        png_raw->height = data->jobs[job_idx]->height;
        png_raw->width = data->jobs[job_idx]->width;
        png_raw->pixel_size = data->jobs[job_idx]->pixel_size;
        png_raw->buf = data->jobs[job_idx]->dest_raw;
        png_raw->info_ptr = data->jobs[job_idx]->info_ptr;
        write_png(data->jobs[job_idx]->dest_name, png_raw);
    }

    free(png_raw); // Free the allocated memory for png_raw
    pthread_exit(NULL);
}

void write_jobs_output_files(PROCESSING_JOB **jobs) {
    int count = 0;
    while (jobs[count] != NULL) {
        count++;
    }

    int num_threads = 9;
    pthread_t threads[num_threads];
    WriteThreadData thread_data[num_threads];
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    int job_index = 0;

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].jobs = jobs;
        thread_data[i].job_index = &job_index;
        thread_data[i].total_jobs = count;
        thread_data[i].mutex = &mutex;

        pthread_create(&threads[i], NULL, write_job, (void *)&thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&mutex);
}
