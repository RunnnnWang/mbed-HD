#include "hd_model.h"
#include <stdio.h>
#include <stdlib.h>
#include "cJSON.h"
#include <time.h>




// Function to process and store the 2D array data
float** process_2d_array(cJSON *json, int *rows, int *cols) {
    // Check if the JSON is an array
    if (!cJSON_IsArray(json)) {
        fprintf(stderr, "JSON data is not an array\n");
        return NULL;
    }

    // Get the number of rows
    *rows = cJSON_GetArraySize(json);
    if (*rows == 0) {
        fprintf(stderr, "Empty data array\n");
        return NULL;
    }

    // Get the number of columns (assuming all rows have the same number of columns)
    cJSON *first_row = cJSON_GetArrayItem(json, 0);
    if (!cJSON_IsArray(first_row)) {
        fprintf(stderr, "First row is not an array\n");
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

void shuffle(float **array1, int *array2, int n, unsigned int seed) {
    if (n > 1) {
        srand(seed);
        for (int i = 0; i < n - 1; i++) {
            int j = i + rand() / (RAND_MAX / (n - i) + 1);
            float *temp1 = array1[j];
            array1[j] = array1[i];
            array1[i] = temp1;

            int temp2 = array2[j];
            array2[j] = array2[i];
            array2[i] = temp2;
        }
    }
}


void test_train_split(float** all_data, int* all_label, int train_amount, int test_amount, int data_amount, int input_dim, int random_state, float** X_train, float ** X_test, int* y_train, int* y_label){
    shuffle(all_data, all_label, train_amount, random_state);
    for(int i = 0; i < train_amount; i ++){
        X_train[i] = all_data[i];
        for(int j = 0; j < input_dim; j ++){
            X_train[i][j] = all_data[i][j];
        }
        
        y_train[i] = all_label[i];
    }

    for(int i = train_amount; i < data_amount; i++){
        X_test[i] = all_data[i];
          for(int j = 0; j < input_dim; j ++){
            X_test[i][j] = all_data[i][j];
        }
        y_train[i] = all_label[i];
    }
}


int main() {
    const char *filename = "data.json";
    const char *labelfile = "label.json";

    char *json_string = read_file(filename);
    char *label_string = read_file(labelfile);
    if (json_string == NULL || label_string == NULL) {
        return 1;
    }


    cJSON *json = parse_json(json_string);
    cJSON *label_json = parse_json(label_string);
    if (json == NULL || label_json == NULL) {
        free(json_string);
        free(label_json);
        return 1;
    }

    int rows, cols;
    int size;
    float **all_data = process_2d_array(json, &rows, &cols);
    int *all_label = process_1d_array(label_json, &size);

   


    // if (array != NULL && all_label != NULL) {
    //     // Print the 2D array
    //     for (int i = 0; i < rows; i++) {
    //         for (int j = 0; j < cols; j++) {
    //             printf("array[%d][%d] = %lf\n", i, j, array[i][j]);
    //         }
    //     } 
    //     printf("label 1%d", all_label[0]);
    // }


    int data_size = 2280;
    int data_in_dim = 256;
    int data_out_dim = 10000;
    float test_size = 0.3;
    int train_amount = (1-test_size)*data_size;
    int test_amount = test_size*data_size;

    unsigned int random_state = 42;  // Fixed seed for reproducibility

    

    


    float X_train[train_amount][data_in_dim];
    float X_test[test_amount][data_in_dim];
    char y_train[train_amount];
    char y_test[test_amount];

    // float** X_test = (float**) malloc(sizeof(float*)*test_amount);
    // for(int i = 0; i < train_amount; i ++){
    //     X_train[i] = (float*) malloc(sizeof(float)*data_in_dim);
    // }
    // for(int i = 0; i < test_amount; i ++){
    //     X_test[i] = (float*) malloc(sizeof(float)*data_in_dim);
    // }

    // int* y_train = (int*) malloc(sizeof(int)*train_amount);
    // int* y_test = (int*) malloc(sizeof(int)*test_amount);




    test_train_split(all_data, all_label, train_amount, test_amount, data_size, data_size, random_state, X_train, X_test, y_train, y_test);

    //    for (int i = 0; i < train_amount; i++) {
    //         for (int j = 0; j < data_in_dim; j++) {
    //             printf("array[%d][%d] = %lf\n", i, j, X_train[i][j]);
    //             printf("array[%d] = %lf\n", i, y_train[i]);
    //         }} 

    hdModel* model = malloc(sizeof(hdModel));
    init_hd_model(model, X_train, X_test, y_train, y_test, data_in_dim, 10000, (float)0.1, test_size, data_size);

    train(model);
    test(model);


    for (int i = 0; i < train_amount; i++) {
            free(X_train[i]);
        }
    for (int i = 0; i < test_amount; i++) {
            free(X_test[i]);
        }
    free(X_train);
    free(X_test);
    free(y_train);
    free(y_test);
    
    // Free the 2D array
    // for (int i = 0; i < rows; i++) {
    //         free(all_data[i]);
    //     }
    free(all_data);
    free(all_label);
    // Clean up
    cJSON_Delete(json);
    cJSON_Delete(label_json);
    free(json_string);
    free(label_string);
    return 0;
}