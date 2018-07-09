#include "Cosine-Transform.h"
#include <iostream>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265359

using namespace std;

double ***f;
int height = 8, width = 8, channels = 3;

void init(){
    f = (double***) malloc(height * sizeof(double**));

    for(int i=0; i<height; ++i){
        f[i] = (double**)malloc(width * sizeof(double*));
        for(int j=0; j<width; ++j){
            f[i][j] = (double*)malloc(channels * sizeof(double));
        }
    }
}

void cosine_transform(double* img, double* img_out){
    init();

    int indice;
    double division_x, division_y;
    double C_u, C_v;
    double sum_r, sum_g, sum_b;

    for(int u=0; u<height; u++){
        for(int v=0; v<width; v++){
            sum_r = 0.0; sum_g = 0.0; sum_b = 0.0;
            for(int x=0; x<height; x++){
                for(int y=0; y<width; y++){
                    indice = (x*width*3)+(y*3);

                    division_x =  ((2*x + 1) * u * PI) / 16.0;
                    division_y =  ((2*y + 1) * v * PI) / 16.0;

                    sum_r += img[indice]   * cos(division_x) * cos(division_y);
                    sum_g += img[indice+1] * cos(division_x) * cos(division_y);
                    sum_b += img[indice+2] * cos(division_x) * cos(division_y);
                }
            }

            if(u == 0)  C_u = 1.0 / sqrt(2);
            else    C_u = 1.0;

            if(v == 0)  C_v = 1.0 / sqrt(2);
            else    C_v = 1.0;

            f[u][v][0] = sum_r * (C_u / 2.0) * (C_v / 2.0);
            f[u][v][1] = sum_g * (C_u / 2.0) * (C_v / 2.0);
            f[u][v][2] = sum_b * (C_u / 2.0) * (C_v / 2.0);

            indice = (u*width*3)+(v*3);

            img_out[indice]   = f[u][v][0];
            img_out[indice+1] = f[u][v][1];
            img_out[indice+2] = f[u][v][2];
        }
    }
}

void inverse_cosine_transform(double* f_in, double* img){
    int indice;
    double division_x, division_y;
    double C_u, C_v;
    double sum_r, sum_g, sum_b;

    for(int x=0; x<height; x++){
        for(int y=0; y<width; y++){
            sum_r = 0.0; sum_g = 0.0; sum_b = 0.0;
            for(int u=0; u<height; u++){
                for(int v=0; v<width; v++){
                    indice = (u*width*3)+(v*3);

                    f[u][v][0] = f_in[indice];
                    f[u][v][1] = f_in[indice+1];
                    f[u][v][2] = f_in[indice+2];

                    division_x =  ((2*x + 1) * u * PI) / 16.0;
                    division_y =  ((2*y + 1) * v * PI) / 16.0;

                    if(u == 0)  C_u = 1.0 / sqrt(2);
                    else    C_u = 1.0;

                    if(v == 0)  C_v = 1.0 / sqrt(2);
                    else    C_v = 1.0;

                    sum_r += f[u][v][0] * (C_u / 2.0) * (C_v / 2.0) * cos(division_x) * cos(division_y);
                    sum_g += f[u][v][1] * (C_u / 2.0) * (C_v / 2.0) * cos(division_x) * cos(division_y);
                    sum_b += f[u][v][2] * (C_u / 2.0) * (C_v / 2.0) * cos(division_x) * cos(division_y);
                }
            }
            indice = (x*width*3)+(y*3);

            img[indice]   = sum_r;
            img[indice+1] = sum_g;
            img[indice+2] = sum_b;
        }
    }
}
