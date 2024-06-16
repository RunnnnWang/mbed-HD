float calculate_standard_deviation(float* arr, int size){
    float mean = calculate_mean(arr, size);
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += pow(arr[i] - mean, 2);
    }
    return sqrt(sum / (size - 1)); 
}



float calculate_mean(float* arr, int size){
    float sum = 0.0;
    for(int i = 0; i < size; i ++){
        sum += arr[i];
    }
    return sum/size;
}


#include <hd_model.h>
#include <stdio.h>


void init_hd_model(hdModel* hd_model, float X_train[TRAIN_AMOUNT][DATA_IN_DIM], float X_test[TEST_AMOUNT][DATA_IN_DIM], char y_train[TRAIN_AMOUNT], char y_test[TEST_AMOUNT] , int in_dim, int out_dim, float lr, float split_size, int data_amount){ //X_test and y_test contains the whole training set, not single data point
    
    
    for(int i = 0; i < TRAIN_AMOUNT; i ++){
        int mean = 0;
        for(int j = 0; j < DATA_IN_DIM; j ++){
            mean += X_train[i][j]; //doing fit at the same time
        }
        hd_model->X_train[i][j] = X_train[i][j];
        hd_model->y_train[i] = y_train[i];
    }

    for(int i = 0; i < TEST_AMOUNT; i ++){
        for(int j = 0; j < DATA_IN_DIM; j ++){
            hd_model->X_test[i][j] = X_test[i][j];
        }
        hd_model->y_test[i] = y_test[i];
    }


    fit(X_train, in_dim, train_amount);
    fit(X_test, in_dim, test_amount);
    
    hd_model->in_dim = in_dim;
    hd_model->out_dim = out_dim;
    hd_model->lr = lr;
    hd_model->split_size = split_size;
    hd_model->data_amount = 2280;

    hd_model->class_size = 12;

    lrp* linear_rp = malloc(sizeof(lrp));
    init_lrp(linear_rp, in_dim, out_dim);
    hd_model->encoder = linear_rp;
    
    //class_hvs = 12 * out_dim 
    hd_model->class_hvs = (char**) malloc(sizeof(char)*12); //hard coding the number of classes
    for(int i = 0; i < 12; i ++){ //hard coding the number of classes
        hd_model->class_hvs[i] = (char*) malloc(sizeof(char)*out_dim);
        for(int j = 0; j < out_dim; j ++){

            hd_model->class_hvs[i][j] = 1;
        }
    }

    hd_model->train_encs = (char**) malloc(sizeof(char*)*train_amount);
    for(int k = 0; k < train_amount; k ++){
        hd_model->train_encs[k] = (char*) malloc(sizeof(char)*out_dim);
    }

}

void train(hdModel* model){
    int n_train = (1-model->split_size)*(model->data_amount);
    for(int i = 0; i < n_train; i ++){
        // if(i != 21){
        //     continue;
        // }
        float* sample = (model->X_train)[i];
        char label = model->y_train[i];
        char* enc = encode(model->encoder, sample);
        float* cos_similarity = (float*) malloc(sizeof(float)*(model->class_size)); //hard coding with number of classes
        //cosine similairty: enc(d*1) * (n*d) --> (1*d)* (d*n) ---> (n*1) ??
        cosine_similarity(cos_similarity, model, enc);
        //printf("i %d \n", i); //
        char index_pred = max_index(cos_similarity, model->class_size);
        if(index_pred != label){
            add_enc(model->class_hvs[label], enc, model->out_dim);
            subtract_enc(model->class_hvs[index_pred], enc, model->out_dim);
        }
        //append the enc to the train encoding
        for(int j = 0; j < model->out_dim; j ++){
            model->train_encs[i][j] = enc[j];
        }
        //append_enc(model->train_encs, enc, i, model->out_dim);
    }
}

void test(hdModel* hd_model){
    int n_test = (hd_model->data_amount)*(hd_model->split_size);
    float correct_count = 0.0; 
    for(int i = 0; i < n_test; i++){
        float* sample = hd_model->X_test[i];
        char label = hd_model->y_test[i];
        char* enc = encode(hd_model->encoder, sample);
        float* cos_similarity = (float*) malloc(sizeof(float)*(hd_model->class_size)); //hard coding with number of classes
        cosine_similarity(cos_similarity, hd_model, enc);
        char index_pred = max_index(cos_similarity, hd_model->class_size);
        if(index_pred == label){
            correct_count += 1;
        }
    }
    printf("accuracy score: %f", correct_count/n_test);
}

void retrain(hdModel* hd_model){
    int train_length =  hd_model->data_amount*hd_model->split_size;
    int count = 0; 
    for(int e = 0; e < 3; e++){
        count = 0;
        for(int i = 0; i < train_length; i ++){
            char* enc = hd_model->train_encs[i];
            char label = hd_model->y_train[i];
            float* cos_similarity = (float*) malloc(sizeof(float)*(hd_model->class_size)); //hard coding with number of classes
            cosine_similarity(cos_similarity, hd_model, enc);
            char index_pred = max_index(cos_similarity, hd_model->class_size);
            if(index_pred != label){
                add_enc(hd_model->class_hvs[label], enc, hd_model->out_dim);
                subtract_enc(hd_model->class_hvs[index_pred], enc, hd_model->out_dim);
                count += 1;
            }
        }
        //printf("count %d", count);
    }
}

void free_hd_model(hdModel* hd_model){
    int train_n = hd_model->data_amount * hd_model->split_size;
    int test_n = hd_model->data_amount * (1-hd_model->split_size);
    for(int i = 0; i < train_n; i ++){
        free(hd_model->X_train[i]);
    }
    for(int i = 0; i < test_n; i ++){
        free(hd_model->X_test[i]);
    }
    free(hd_model->X_train);
    free(hd_model->X_test);

    free(hd_model->y_test);
    free(hd_model->y_train);
    
    free_lrp(hd_model->encoder);
    
    for(int i = 0; i < hd_model->class_size; i ++){
        free(hd_model->class_hvs[i]);
    }
    free(hd_model->class_hvs);

    for(int i = 0; i < train_n; i ++){
        free(hd_model->train_encs[i]);
    }
    free(hd_model->train_encs);

    free(hd_model);
    
}

void fit(float input_feature[][DATA_IN_DIM]){
    for(int i = 0; i < DATA_SIZE; i ++){
        float mean = 0; 
        for(int j = 0; j < DATA_IN_DIM; j ++){
            mean += input_feature[i][j];
        }

        float standard_deviation = calculate_standard_deviation(feature, in_dim);
        for(int j = 0; j < in_dim; j ++){
            feature[j] = (feature[j]-mean)/standard_deviation;
        }
    }
}





char dot_product(char* vec1, char* vec2, int length){
    //cosine similairty: enc(d*1) * (n*d) --> (1*d)* (d*n) ---> (n*1)
    float dot = 0.0; 
    for(int i = 0; i < length; i ++){
        dot += vec1[i] * vec2[i];
    }
    return dot;
}

// char magnitude(char* vec1, int length){
//     char magnitude = 0;
//     for(int i = 0; i < length; i ++){
//         magnitude += vec1[i]*vec1[i];
//     }
//     return sqrt(magnitude);
// }


void cosine_similarity(float* cosine_similarity, hdModel* model, char* enc){
    for (int i = 0; i < model->class_size; i ++){
        float similarity = 0.0;
        int dot = 0; 
        for(int j = 0; j < model->out_dim; j ++){
            dot += model->class_hvs[i][j] + enc[j];

        }
        //float dot = dot_product(enc, model->class_hvs[i], model->out_dim);

        //char magnitude_enc = magnitude(enc, model->out_dim);

        int magnitude_enc = 0;
        for(int j = 0; j <  model->out_dim; j ++){
        magnitude_enc += enc[j]*enc[j]; }
        magnitude_enc = sqrt(magnitude_enc);

        int magnitude_class_vec = 0;
        for(int j = 0; j <  model->out_dim; j ++){
        magnitude_class_vec += (model->class_hvs[i][j])*(model->class_hvs[i][j]); }
        magnitude_class_vec = sqrt(magnitude_class_vec);

        //char magnitude_class_vec = magnitude(model->class_hvs[i], model->out_dim);
        cosine_similarity[i] = dot/(magnitude_enc*magnitude_class_vec);
    }
}

int max_index(float* vec, int length){
    int max = 0;
    for(int i = 0; i < length; i++ ){
        if(vec[i] > vec[max]){
            vec[max] = vec[i];
        }
    }
    return max;
}

void add_enc(char* vec1, char* enc, int length){
    for(int i = 0; i < length; i ++){
        vec1[i] = vec1[i] + enc[i];
    }
}

void subtract_enc(char* vec1, char* enc, int length){
    for(int i = 0; i < length; i ++){
        vec1[i] = vec1[i] - enc[i];
    }
}

void append_enc(char** total_train_encs, char* enc, int index, int length){
    for(int i = 0; i < length; i ++){
        total_train_encs[index][i] = enc[i];
    }
}