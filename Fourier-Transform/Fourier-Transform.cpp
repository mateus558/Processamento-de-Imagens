#include "Fourier-Transform.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include <complex.h>

#define PI 3.14159265359

using namespace std;

complex<double> ***f;
complex<double> ***f_filtered;
int filter = 0;
int height_aux, width_aux, channels = 3;

void init(){
    f = (complex<double>***) malloc(height_aux * sizeof(complex<double>**));

    for(int i=0; i<height_aux; ++i){
        f[i] = (complex<double>**)malloc(width_aux * sizeof(complex<double>*));
        for(int j=0; j<width_aux; ++j){
            f[i][j] = (complex<double>*)malloc(channels * sizeof(complex<double>));
        }
    }
}

void fourier_transform(int* img, int height, int width, double* img_real, double* img_imag, int filter, int radius_1, int radius_2){
    height_aux = height;
    width_aux = width;

    init();

    complex<double> i = csqrt(-1);
    complex<double> exponential;
    complex<double> sumR = {0.0, 0.0}, sumG = {0.0, 0.0}, sumB = {0.0, 0.0};
    complex<double> aux;

    int indice;
    float M = (height * 1.0), N = (width * 1.0);
    complex<double> division = {1 / (sqrt(M*N)), 1 / (sqrt(M*N))};

    for(int u=0; u<height; u++){
        for(int v=0; v<width; v++){
            sumR = {0.0, 0.0}; sumG = {0.0, 0.0}; sumB = {0.0, 0.0};
            for(int x=0; x<height; x++){
                for(int y=0; y<width; y++){
                    aux = {2*PI*((u*x/M) + (v*y/N)), 0.0};
                    exponential = exp(-i*aux);

                    sumR += ((complex<double>)img[(x*width*3)+(y*3)]  ) * exponential;
                    sumG += ((complex<double>)img[(x*width*3)+(y*3)+1]) * exponential;
                    sumB += ((complex<double>)img[(x*width*3)+(y*3)+2]) * exponential;
                }
            }

            f[u][v][0] = sumR;
            f[u][v][1] = sumG;
            f[u][v][2] = sumB;

            indice = (u*width*3)+(v*3);

            img_real[indice]   = f[u][v][0].real();
            img_real[indice+1] = f[u][v][1].real();
            img_real[indice+2] = f[u][v][2].real();

            img_imag[indice]   = f[u][v][0].imag();
            img_imag[indice+1] = f[u][v][1].imag();
            img_imag[indice+2] = f[u][v][2].imag();
        }
    }

    if(filter != 0){
        ideal_pass_filter(img_real, img_imag, filter, radius_1, radius_2);
    }
}

void inverse_fourier_transform(double* img_real, double* img_imag, int* img){
    complex<double> i = csqrt(-1);
    complex<double> exponential;
    complex<double> sumR = {0.0, 0.0}, sumG = {0.0, 0.0}, sumB = {0.0, 0.0};
    complex<double> aux;

    int indice, indice2;
    float M = (height_aux * 1.0), N = (width_aux * 1.0);
    float division = 1 / ((M*N));

    for(int x=0; x<height_aux; x++){
        for(int y=0; y<width_aux; y++){
            sumR = {0.0, 0.0}; sumG = {0.0, 0.0}; sumB = {0.0, 0.0};
            for(int u=0; u<height_aux; u++){
                for(int v=0; v<width_aux; v++){
                    indice = (u*width_aux*3)+(v*3);

                    f[u][v][0] = {img_real[indice]  , img_imag[indice]  };
                    f[u][v][1] = {img_real[indice+1], img_imag[indice+1]};
                    f[u][v][2] = {img_real[indice+2], img_imag[indice+2]};

                    aux = 2*PI*((u*x/M) + (v*y/N));
                    exponential = exp(i*aux);

                    sumR += f[u][v][0] * exponential;
                    sumG += f[u][v][1] * exponential;
                    sumB += f[u][v][2] * exponential;
                }
            }
            indice2 = (x*width_aux*3)+(y*3);

            img[indice2]   = sumR.real()*division;
            img[indice2+1] = sumG.real()*division;
            img[indice2+2] = sumB.real()*division;
        }
    }
}

void ideal_pass_filter(double* img_real, double* img_imag, int filter, int radius_1, int radius_2){
	if(filter == 1){
		if(radius_1 > radius_2){
			int aux = radius_1;
			radius_1 = radius_2;
			radius_2 = aux;
		}
	}

	int center_x = height_aux / 2;
	int center_y = width_aux / 2;
	double distance;
	int x;
	for(int i=0; i<height_aux; i++){
        for(int j=0; j<width_aux; j++){
            distance = sqrt((i - center_x)*(i - center_x) + (j - center_y)*(j - center_y));
            if (   (filter == 1 && (distance <= radius_1 || distance >= radius_2))
                || (filter == 2 && distance >= radius_1)
                || (filter == 3 && distance <= radius_1)){

                x = (i*width_aux*3)+(j*3);

				img_real[x]   = 0.0;
				img_real[x+1] = 0.0;
				img_real[x+2] = 0.0;

				img_imag[x]   = 0.0;
				img_imag[x+1] = 0.0;
				img_imag[x+2] = 0.0;

            }
        }
	}
}
