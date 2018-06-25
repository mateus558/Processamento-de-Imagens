#include "Fourier-Transform.h"
#include "main.h"

extern "C"{
    void Fourier_transform(int* img, int height, int width, double* img_real, double* img_imag, int filter, int radius_1, int radius_2){
        fourier_transform(img, height, width, img_real, img_imag, filter, radius_1, radius_2);
    }
    void Inverse_fourier_transform(double* img_real, double* img_imag, int* img){
        inverse_fourier_transform(img_real, img_imag, img);
    }
}
