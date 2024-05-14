#include "header.h"

void PictureHost_FILTER(png_byte *h_In, png_byte *h_Out, int h, int w, float *h_filt)
{
    float out;
    png_byte b;

    for (int Row = 2; Row < h - 2; Row++)
        for (int Col = 2; Col < w - 2; Col++)
        {
            for (int color = 0; color < 3; color++)
            {
                out = 0;
                for (int i = -2; i <=2; i++)
                    for(int j=-2; j<=2; j++)
                        out += h_filt[(i+2)*5+j+2] * h_In[((Row+i)*w+(Col+j)) * 3 + color];
                b = (png_byte)fminf(fmaxf(out, 0), 255);
                h_Out[(Row*w+Col) * 3 + color] = b;
            }
        }
}

void execute_jobs_cpu(PROCESSING_JOB **jobs)
{
    int count=0;
    float *h_filter;
    while (jobs[count]!=NULL)
    {

        h_filter = getAlgoFilterByType(jobs[count]->processing_algo);
        PictureHost_FILTER(jobs[count]->source_raw, jobs[count]->dest_raw,
                           jobs[count]->height, jobs[count]->width, h_filter);
        count++;
    }

}
