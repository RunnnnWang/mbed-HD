#include <linear_random_projection.h>

void init_lrp(lrp* lrp, int in_dim, int out_dim){
    lrp->in_dim = in_dim;
    lrp->out_dim = out_dim;
    init_projection(lrp);
}

void init_projection(lrp* lrp){ //D * n
    lrp->projection = (char**) malloc(lrp->in_dim * sizeof(char));
    for (int i = 0; i < lrp->in_dim; i ++) {
        lrp->projection[i] = (char*) malloc(lrp->out_dim * sizeof(char)); //should I use malloc??
        for (int j = 0; j < lrp->out_dim; j ++) {
        lrp->projection[i][j] = sign(generate_normal());
    }
    }
}

char* encode(lrp* lrp, float* x){ //x.shape = n * 1
    char* enc = (char*) malloc(lrp->out_dim * sizeof(char));
    for (int i = 0; i < lrp->out_dim; i ++) {
        for (int j = 0; j < lrp->in_dim; j ++) {
            enc[i] += sign(lrp->projection[i][j]+ x[j]);
        }
    }
    return enc;
}

void free_lrp(lrp *lrp){
    for (int i = 0; i < lrp->in_dim; i ++) {
        free(lrp->projection[i]);
        }
    free(lrp->projection);
    free(lrp);
}

//-----------------helper method------------------------------------------------
float generate_normal() {
    float u1, u2, z0;
    
    // Generate u1 and u2, two uniform random numbers between 0 and 1
    u1 = rand() / (RAND_MAX + 1.0);
    u2 = rand() / (RAND_MAX + 1.0);
    
    // Box-Muller transform
    z0 = sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
    
    return z0; // return the normally distributed random number
}

char sign(float num){
    if (num > 0) {
        return 1;
    }
    if (num < 0) {
        return -1;
    }
    if (num == 0) {
        return 0;
    }
}


