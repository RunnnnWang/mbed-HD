#include <math.h>
#include <linear_random_projection.h>

#ifndef HD_MODEL_H
#define HD_MODEL_H

typedef struct {
    float** X_train;
    float** X_test; //are they even all char pointer? what is the type?
    char* y_train;
    char* y_test;
    int in_dim;
    int out_dim;
    float lr;
    float split_size;
    int data_amount;
    lrp* encoder;
    char** class_hvs;
    char** train_encs;
    int class_size;

} hdModel;

void init_hd_model(hdModel* hd_model, float* X_train, float* X_test, float* y_train, float* y_test, int in_dim, int out_dim, float lr, float split_size);
void train(hdModel* hd_model);
void test(hdModel* hd_model);
void retrain(hdModel* hd_model);
void free_hd_model(hdModel* hd_model);


//----------------------------------------helper method------------------------------------
void fit(float** input_feature, int in_dim);
void transform(float** input_feature);

float calculate_mean(float* arr, int size);
float calculate_standard_deviation(float* arr, int size);
float dot_product(float* vec1, float* vec2, int length);
float magnitude(float* vec1, int length);


void cosine_similarity(float* cosine_similarity, hdModel* model, float* enc);
int max_index(float* vec, int length);
void add_enc(float* vec1, char* enc, int length);
void subtract_enc(float* vec1, char* enc, int length);
void append_enc(char** train_encs, char* enc, int length);



#endif


