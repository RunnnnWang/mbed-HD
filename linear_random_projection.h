#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>

#ifndef LINEAR_RANDOM_PROJECTION_H //lrp
#define LINEAR_RANDOM_PROJECTION_H

typedef struct {
    int in_dim;
    int out_dim;
    char** projection;
} lrp;

void init_lrp(lrp* lrp, int in_dim, int out_dim);
void free_lrp(lrp* lrp);
void init_projection(lrp* lrp);
char* encode(lrp* lrp, float* x);

//-----------------------------------helper method-----------------------------
float generate_normal();
char sign(float num);


#endif




