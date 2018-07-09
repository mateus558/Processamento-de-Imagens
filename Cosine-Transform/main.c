#include "Cosine-Transform.h"
#include "main.h"

extern "C"{
    void Cosine_transform(double* img, double* img_out){
        cosine_transform(img, img_out);
    }
    void Inverse_cosine_transform(double* f_in, double* img){
        inverse_cosine_transform(f_in, img);
    }
}
