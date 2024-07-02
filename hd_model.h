#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <time.h>

#ifndef HD_MODEL_H
#define HD_MODEL_H

#define DATA_SIZE 2280
#define TRAIN_AMOUNT 1596
//#define TRAIN_AMOUNT 5
#define TEST_AMOUNT 684

#define DATA_IN_DIM 256
#define DUMMY_DATA_IN_DIM 10
#define DATA_OUT_DIM 10000
#define CLASS_AMOUNT 12


#define SPLIT 0.3
#define LEARNING_RATE 0.1

typedef struct {
    int class_hvs[CLASS_AMOUNT][DATA_OUT_DIM];
    int highest_class_hvs[CLASS_AMOUNT][DATA_OUT_DIM];
    float train_encs[TRAIN_AMOUNT][DATA_OUT_DIM];
    float X_train[TRAIN_AMOUNT][DATA_IN_DIM];
    float X_test[TEST_AMOUNT][DATA_IN_DIM]; 
    int y_train[TRAIN_AMOUNT];
    int y_test[TEST_AMOUNT];
    int projection[DATA_OUT_DIM][DATA_IN_DIM]; //IN_dim = # rows, out dim = # columns; so d * n
} hdModel;

void init_hd_model(hdModel* hd_model, float** all_data, int* all_label, int sh);
void train(hdModel* hd_model);
float test(hdModel* hd_model, int seed, int use_best_class_hv);
int retrain(hdModel* hd_model);


//--------------------------------------linear random projection method--------------------------------------
void init_lrp(hdModel* hd_model);
void encode(hdModel* hd_model, int index);
    //-----------------------------------helper method-----------------------------
    float generate_normal_random_float();
    int sign(float num);



//----------------------------------------helper method------------------------------------
void shuffle(float **array1, int *array2, int n, unsigned int seed);
void transform(float** input_feature);



void append_enc(int** total_train_encs, int* enc, int index, int length);



#endif


