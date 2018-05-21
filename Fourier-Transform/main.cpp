#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <complex>
#include <omp.h>

#define PI 3.14159265359

using namespace std;

complex<double> ***f;
double ***img;
int height, width, channels, algorithm;


void fourier_transform(){
    complex<double> i; i.real() = 0.0 ; i.imag() = 1.0;
    complex<double> exponential;

    int M = height, N = width;

    # pragma omp for
    for(int u=0; u<height; u++){
        for(int v=0; v<width; v++){
        complex<double> sumR = 0.0, sumG = 0.0, sumB = 0.0;
            sumR = 0.0; sumG = 0.0; sumB = 0.0;
            for(int x=0; x<height; x++){
                for(int y=0; y<width; y++){
                    complex<double> aux = 2*PI*((u*x/M) + (v*y/N));
                    exponential = exp(-i*aux);
                    sumR += img[x][y][0] * exponential;
                    sumG += img[x][y][1] * exponential;
                    sumB += img[x][y][2] * exponential;
                }
            }
            f[u][v][0] = sumR; f[u][v][1] = sumG; f[u][v][2] = sumB;
        }
    }
}

void fourier_transform_inverse(){
    complex<double> sumR = 0.0, sumG = 0.0, sumB = 0.0;
    complex<double> i = sqrt(-1);
    complex<double> exponential;

    int M = height, N = width;

    # pragma omp for
    for(int x=0; x<height; x++){
        for(int y=0; y<width; y++){
            sumR = 0.0; sumG = 0.0; sumB = 0.0;
            for(int u=0; u<height; u++){
                for(int v=0; v<width; v++){
                    exponential = exp(i*((complex<double>)(2*PI*((u*x/M) + (v*y/N)))));
                    sumR += f[u][v][0] * exponential;
                    sumG += f[u][v][1] * exponential;
                    sumB += f[u][v][2] * exponential;
                }
            }
            img[x][y][0] = sumR.real()/(M*N); img[x][y][1] = sumG.real()/(M*N); img[x][y][2] = sumB.real()/(M*N);
        }
    }
}


/**
* Paramns:
*   - (int) Algorithm selected:
*           - 0: fourier_transform_inverse()
*           - 1: fourier_transform()
*   - (int) Height
*   - (int) Width
*   - (int) Channels
*   - (char*) File Name
*
**/

int main(int argc, char *argv[]){
    cout << argc << endl;
    if(argc != 6){
        cerr << "Too few arguments to run" << endl;
        exit(1);
    }

    char* file_name, file_aux;
    FILE* file;

    algorithm = atoi(argv[1]);
    height    = atoi(argv[2]);
    width     = atoi(argv[3]);
    channels  = atoi(argv[4]);
    file_name = argv[5];

    //cout << algorithm << " "  << height << " " << width << " " << channels << " " << file_name << endl;

    file = fopen(file_name, "r");

    img = (double***) malloc(height * sizeof(double**));

    for(int i=0; i<height; ++i){
        img[i] = (double**)malloc(width * sizeof(double*));
        for(int j=0; j<width; ++j){
            img[i][j] = (double*)malloc(channels * sizeof(double));
        }
    }

    f = (complex<double>***) malloc(height * sizeof(complex<double>**));

    for(int i=0; i<height; ++i){
        f[i] = (complex<double>**)malloc(width * sizeof(complex<double>*));
        for(int j=0; j<width; ++j){
            f[i][j] = (complex<double>*)malloc(channels * sizeof(complex<double>));
        }
    }

    int value;
    file_aux = fscanf(file, "%d", &value);

    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            for(int k=0; k<channels; k++){
                if(file_aux != EOF){
                    img[i][j][k] = value;
                    file_aux = fscanf(file, "%d", &value);
                }
            }
        }
    }

    fclose(file);

    if(algorithm)
        fourier_transform();
    else
        fourier_transform_inverse();


    file_name = "img_array_real.txt";
    file = fopen(file_name, "w");

    for(int u=0; u<height; u++){
        for(int v=0; v<width; v++){
            for(int k=0; k<channels; k++){
                fprintf(file, "%d\n", (int)f[u][v][k].real());
            }
        }
    }

    fclose(file);

    file_name = "img_array_imag.txt";
    file = fopen(file_name, "w");

    for(int u=0; u<height; u++){
        for(int v=0; v<width; v++){
            for(int k=0; k<channels; k++){
                fprintf(file, "%d\n", (int)f[u][v][k].imag());
            }
        }
    }

    fclose(file);

    return 0;
}
