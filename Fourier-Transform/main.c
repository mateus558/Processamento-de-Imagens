#include "Fourier-Transform.h"
#include "main.h"

extern "C"{
    void Fourier_transform(int* img, int height, int width, double* img_real, double* img_imag){
        fourier_transform(img, height, width, img_real, img_imag);
    }
    void Inverse_fourier_transform(double* img_real, double* img_imag, int* img){
        inverse_fourier_transform(img_real, img_imag, img);
    }/*
    void Ideal_high_pass_filter(int radius, double* img_real, double* img_imag){
        ideal_high_pass_filter(radius, img_real, img_imag);
    }
    void Ideal_low_pass_filter(int radius, double* img_real, double* img_imag){
        ideal_low_pass_filter(radius, img_real, img_imag);
    }*/
}



