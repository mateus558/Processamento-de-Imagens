#include <stdlib.h>

extern "C"{
    void Fourier_transform(int* img, int height, int width, double* img_real, double* img_imag);
    void Inverse_fourier_transform(double* img_real, double* img_imag, int* img);
   /* void Ideal_high_pass_filter(int radius, double* img_real, double* img_imag);
    void Ideal_low_pass_filter(int radius, double* img_real, double* img_imag);*/
}
