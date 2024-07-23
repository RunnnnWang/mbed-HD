#include "hd_model.h"
#include <stdio.h>
#include <stdlib.h>
#include "cJSON.h"
#include <time.h>

//--------------------------------------------process data function------------------------------
// Function to process and store the 2D array data
float** process_2d_array(cJSON *json, int *rows, int *cols) {
    // Check if the JSON is an array
    if (!cJSON_IsArray(json)) {
        fprintf(stderr, "JSON data is not an array\n");
        printf("in 1st if\n");
        return NULL;
    }

    // Get the number of rows
    *rows = cJSON_GetArraySize(json);
    if (*rows == 0) {
        fprintf(stderr, "Empty data array\n");
        printf("in 2st if\n");
        return NULL;
    }

    // Get the number of columns (assuming all rows have the same number of columns)
    cJSON *first_row = cJSON_GetArrayItem(json, 0);
    if (!cJSON_IsArray(first_row)) {
        fprintf(stderr, "First row is not an array\n");
        printf("in 3st if\n");
        return NULL;
    }

    *cols = cJSON_GetArraySize(first_row);
    
    // Allocate memory for the 2D array
    float **array = malloc(*rows * sizeof(float*));
    for (int i = 0; i < *rows; i++) {
        array[i] = malloc(*cols * sizeof(float));
        cJSON *row = cJSON_GetArrayItem(json, i);
        if (!cJSON_IsArray(row)) {
            fprintf(stderr, "Row %d is not an array\n", i);
            continue;
        }

        for (int j = 0; j < *cols; j++) {
            cJSON *item = cJSON_GetArrayItem(row, j);
            if (cJSON_IsNumber(item)) {
                array[i][j] = item->valuedouble;
            } else {
                fprintf(stderr, "data[%d][%d] is not a number\n", i, j);
                array[i][j] = 0; // Default value if not a number
            }
        }
    }
    return array;
}


int* process_1d_array(cJSON *json, int *size) {
    if (!cJSON_IsArray(json)) {
        fprintf(stderr, "JSON data is not an array\n");
        return NULL;
    }

    *size = cJSON_GetArraySize(json);
    if (*size == 0) {
        fprintf(stderr, "Empty data array\n");
        return NULL;
    }

    int *array = malloc(*size * sizeof(int));
    for (int i = 0; i < *size; i++) {
        cJSON *item = cJSON_GetArrayItem(json, i);
        if (cJSON_IsNumber(item)) {
            array[i] = item->valueint;
        } else {
            fprintf(stderr, "data[%d] is not a number\n", i);
            array[i] = 0; // Default value if not a number
        }
    }
    return array;
}

char* read_file(const char* filename) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("File opening failed");
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *content = malloc(length + 1);
    if (content) {
        fread(content, 1, length, file);
        content[length] = '\0';
    }

    fclose(file);
    return content;
}

cJSON* parse_json(const char* json_string) {
    cJSON *json = cJSON_Parse(json_string);
    if (json == NULL) {
        const char *error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL) {
            fprintf(stderr, "Error before: %s\n", error_ptr);
        }
        return NULL;
    }
    return json;
}

//--------------------------------main------------------------------------------


int main() {
    float accuracy = 0; 
    int iterations = 10;
    // for(int k = 0; k < iterations; k ++){
    // const char *filename = "dataNew.json";
    // const char *labelfile = "label.json";

    // char *json_string = read_file(filename);
    // char *label_string = read_file(labelfile);
    // if (json_string == NULL || label_string == NULL) {
    //     return 1;
    // }


    // cJSON *json = parse_json(json_string);
    // cJSON *label_json = parse_json(label_string);
    // if (json == NULL || label_json == NULL) {
    //     free(json_string);
    //     free(label_json);
    //     return 1;
    // }

    int rows, cols;
    int size;
    // float **all_data = process_2d_array(json, &rows, &cols);
    // int *all_label = process_1d_array(label_json, &size);



    hdModel* model = malloc(sizeof(hdModel));
    //init_hd_model(model, all_data, all_label, k);

    
    // for(int i = 0; i < 10; i ++){
    //     int count1 = 0;
    //     int countn1 = 0;
    //     for(int j = 0; j < 256; j ++){
    //         if(model->projection[i][j]==1){
    //             count1 += 1;
    //         }
    //         else{
    //             countn1 += 1;
    //         }
    //     }
    //     printf("c1 : %d, cn1: %d", count1, countn1);
    // }


    // const char *projection_filename = "projection.json";
    // char *projection_string =read_file(projection_filename);
    // cJSON *parse_projection = parse_json(projection_string);
    int data_in;
    int data_out;
    // float **projection_matrix = process_2d_array(parse_projection, &data_out, &data_in);
    // // printf("s//...%d, %d\n", data_out, data_in);
    // for (int i = 0; i < DATA_OUT_DIM; i ++) {
    //     for (int j = 0; j < DATA_IN_DIM; j ++) {
    //         model->projection[i][j] = sign(projection_matrix[i][j]);
    //     }
    // }

    const char *Xtrain_filename = "Xtrain.json";
    const char *Xtest_filename = "Xtest.json";
    const char *Ytrain_filename = "Ytrain.json";
    const char *Ytest_filename = "Ytest.json";
    const char *Projfection_filename = "projection.json";
    char *Xtrain_string =read_file(Xtrain_filename);
    char *Xtest_string =read_file(Xtest_filename);
    char *Ytrain_string =read_file(Ytrain_filename);
    char *Ytest_string =read_file(Ytest_filename);
    //char *Projfection_string =read_file(Projfection_filename);
    cJSON *parse_Xtrain = parse_json(Xtrain_string);
    cJSON *parse_Xtest = parse_json(Xtest_string);
    cJSON *parse_Ytrain = parse_json(Ytrain_string);
    cJSON *parse_Ytest = parse_json(Ytest_string);
    //cJSON *parse_Projection = parse_json(Projfection_string);
    float **Xtrain = process_2d_array(parse_Xtrain, &data_out, &data_in);
    float **Xtest = process_2d_array(parse_Xtest, &data_out, &data_in);
    int *Ytrain = process_1d_array(parse_Ytrain, &data_in);
    int *Ytest = process_1d_array(parse_Ytest, &data_out);
    //char **Projection = process_2d_array(parse_Projection, &data_out, &data_in);
    // for (int i = 0; i < TRAIN_AMOUNT; i ++) {
    //     for (int j = 0; j < DATA_IN_DIM; j ++) {
    //         model->X_train[i][j] = Xtrain[i][j];
    //     }
    // }
    // for (int i = 0; i < TEST_AMOUNT; i ++) {
    //     for (int j = 0; j < DATA_IN_DIM; j ++) {
    //         model->X_test[i][j] = Xtest[i][j];
    //     }
    // }
    // for (int i = 0; i < TRAIN_AMOUNT; i++){
    //     model->y_train[i] = Ytrain[i];
    // }
    // for (int i = 0; i < TEST_AMOUNT; i++){
    //     model->y_test[i] = Ytest[i];
    // }

    dump_init_hd_model(model, Xtrain, Xtest, Ytrain, Ytest);

    train(model);
    test(model, 0, 0);

    float use_higest_hv = retrain(model);
    printf("%f\n", use_higest_hv);
    accuracy += test(model, 0, use_higest_hv);
    //free the model
    free(model);
  
    //free the original data
    // free(all_data);
    // free(all_label);


    free(Xtrain);
    free(Xtest);
    free(Ytrain);
    free(Ytest);


    cJSON_Delete(parse_Xtrain);
    cJSON_Delete(parse_Xtest);
    cJSON_Delete(parse_Ytrain);
    cJSON_Delete(parse_Ytest);

    //delete reading in the data
    // cJSON_Delete(json);
    // cJSON_Delete(label_json);
    // free(json_string);
    // free(label_string);

    free(Xtrain_string);
    free(Xtest_string);
    free(Ytrain_string);
    free(Ytest_string);
    
    //printf("average accuracy: %f", accuracy/iterations);
    return 0;
}