void fourier_transform(int* img, int height, int width, double* img_real, double* img_imag, int filter, int radius_1, int radius_2);
void inverse_fourier_transform(double* img_real, double* img_imag, int* img);
void ideal_pass_filter(double* img_real, double* img_imag, int filter, int radius_1, int radius_2);
