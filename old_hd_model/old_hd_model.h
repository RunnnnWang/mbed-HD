// #include <math.h>
// #include "linear_random_projection.h"

// #ifndef HD_MODEL_H
// #define HD_MODEL_H

// #define DATA_SIZE 2280
// #define TRAIN_AMOUNT 1596
// #define TEST_AMOUNT 684

// #define DATA_IN_DIM 256
// #define DATA_OUT_DIM 10000
// #define CLASS_AMOUNT 12

typedef struct {
    int in_dim;
    int out_dim;
    float lr;
    float split_size;
    int data_amount;
    int class_size;
    lrp* encoder;
    char class_hvs[CLASS_AMOUNT][DATA_OUT_DIM];
    char train_encs[TRAIN_AMOUNT][DATA_OUT_DIM];
    float X_train[TRAIN_AMOUNT][DATA_IN_DIM];
    float X_test[TEST_AMOUNT][DATA_IN_DIM]; 
    char y_train[TRAIN_AMOUNT];
    char y_test[TEST_AMOUNT];

} hdModel;

void init_hd_model(hdModel* hd_model, float X_train[TRAIN_AMOUNT][DATA_IN_DIM], float X_test[TEST_AMOUNT][DATA_IN_DIM], char y_train[TRAIN_AMOUNT], char y_test[TEST_AMOUNT] , int in_dim, int out_dim, float lr, float split_size, int data_amount);
void train(hdModel* hd_model);
void test(hdModel* hd_model);
void retrain(hdModel* hd_model);
void free_hd_model(hdModel* hd_model);


//----------------------------------------helper method------------------------------------
void fit(float** input_feature, int in_dim, int data_amount);
void transform(float** input_feature);

float calculate_mean(float* arr, int size);
float calculate_standard_deviation(float* arr, int size);
char dot_product(char* vec1, char* vec2, int length);
char magnitude(float* vec1, int length);


void cosine_similarity(float* cosine_similarity, hdModel* model, char* enc);
int max_index(float* vec, int length);
void add_enc(char* vec1, char* enc, int length);
void subtract_enc(char* vec1, char* enc, int length);
void append_enc(char** total_train_encs, char* enc, int index, int length);



#endif