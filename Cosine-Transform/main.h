#include <stdlib.h>

extern "C"{
    void Cosine_transform(int* img, double* img_out);
    void Inverse_cosine_transform(double* f_in, int* img);
}
