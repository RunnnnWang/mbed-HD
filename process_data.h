#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <sys/stat.h>

#ifndef PROCESS_DATA_H
#define PROCESS_DATA_H

typedef struct {
    float** all_data;
    char* all_label;
    int data_size;
    int split_size;
} data;

void init_data(data* data, int data_size, int split_size, char* path_name);
void split_data(data* data);

#endif