#include <stdlib.h>

extern "C"{
    void Fourier_transform(int* img, int height, int width, double* img_real, double* img_imag, int filter, int radius_1, int radius_2);
    void Inverse_fourier_transform(double* img_real, double* img_imag, int* img);
}
