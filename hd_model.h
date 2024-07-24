#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#define _USE_MATH_DEFINES
#include <time.h>

#ifndef HD_MODEL_H
#define HD_MODEL_H

#define DATA_SIZE 3600      //2280  //18000 //3600
#define TRAIN_AMOUNT 2520 //14400 //1596 //2520
//#define TRAIN_AMOUNT 5
#define TEST_AMOUNT 1080  //3600 //1080 //684
#define DATA_IN_DIM 256
// #define DUMMY_DATA_IN_DIM 10
#define DATA_OUT_DIM 2000
#define CLASS_AMOUNT 10  //50 10
#define SPLIT 0.3       //0.2
#define LEARNING_RATE 0.1

typedef struct {
    signed char class_hvs[CLASS_AMOUNT][DATA_OUT_DIM];//8 * 12 * 10000 = 960,000
    // signed char highest_class_hvs[CLASS_AMOUNT][DATA_OUT_DIM];
    signed char train_encs[TRAIN_AMOUNT][DATA_OUT_DIM/8]; // 8 * 1596 * 10000 =127，890,000
    signed char X_train[TRAIN_AMOUNT][DATA_IN_DIM]; // 32 * 1596 * 256 = 13,000, 000
    signed char X_test[TEST_AMOUNT][DATA_IN_DIM];  // 32 * 684 * 256 = 5,603,328
    signed char y_train[TRAIN_AMOUNT]; // 8 * 1596 = 12,768
    signed char y_test[TEST_AMOUNT]; // 8 * 684 = 5,472
    signed char projection[DATA_OUT_DIM][DATA_IN_DIM/8]; //8 * 10000 * 256 = 20,480,000   IN_dim = # rows, out dim = # columns; so d * n
    //167,951,568
    //20,993,946
} hdModel;

void init_hd_model(hdModel* hd_model, float** all_data, int* all_label, int sh);
void dump_init_hd_model(hdModel* hd_model, float** x_train, float** x_test, int* y_train, int* y_test);
void dump_init_hd_model_projection(hdModel* hd_model, float** x_train, float** x_test, int* y_train, int* y_test, float** linear_projection);
void dump_trained_hd_model(hdModel* hd_model, float** x_train, float** x_test, int* y_train, int* y_test, char** linear_projection, char** class_hvs);
void train(hdModel* hd_model);
float test(hdModel* hd_model, int seed, int use_best_class_hv);
float retrain(hdModel* hd_model);


//--------------------------------------linear random projection method--------------------------------------
void init_lrp(hdModel* hd_model);
void dump_lrp(hdModel* hd_model);
void encode(hdModel* hd_model, int index);
    //-----------------------------------helper method-----------------------------
    float generate_normal_random_float();
    int sign(float num);



//----------------------------------------helper method------------------------------------
void shuffle(float **array1, int *array2, int n, unsigned int seed);
void transform(float** input_feature);



void append_enc(int** total_train_encs, int* enc, int index, int length);



#endif


