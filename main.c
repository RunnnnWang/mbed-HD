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

//--------------------------------main------------------------------------------


int main() {
    const char *filename = "dataNew.json";
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



    hdModel* model = malloc(sizeof(hdModel));
    init_hd_model(model, all_data, all_label);
    for(int i = 0; i < 10; i ++){
        int count1 = 0;
        int countn1 = 0;
        for(int j = 0; j < 256; j ++){
            if(model->projection[i][j]==1){
                count1 += 1;
            }
            else{
                countn1 += 1;
            }
        }
        printf("c1 : %d, cn1: %d", count1, countn1);
    }

    train(model);
    test(model);

    retrain(model);
    test(model);
    //free the model
    free(model);
  
    //free the original data
    free(all_data);
    free(all_label);

    //delete reading in the data
    cJSON_Delete(json);
    cJSON_Delete(label_json);
    free(json_string);
    free(label_string);

    return 0;
}