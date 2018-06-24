#include "Cosine-Transform.h"
#include "main.h"

extern "C"{
    void Cosine_transform(int* img, int height, int width, double* img_out){
        cosine_transform(img, height, width, img_out);
    }
    void Inverse_cosine_transform(double* f_in, int* img){
        inverse_cosine_transform(f_in, img);
    }
}
