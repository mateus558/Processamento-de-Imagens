#include <stdlib.h>

extern "C"{
    void Cosine_transform(int* img, int height, int width, double* img_out);
    void Inverse_cosine_transform(double* f_in, int* img);
}
