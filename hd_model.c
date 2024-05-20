#include <hd_model.h>



void init_hd_model(hdModel* hd_model, float** X_train, float** X_test, float* y_train, float* y_test, int in_dim, int out_dim, float lr, float split_size, int data_amount){ //X_test and y_test contains the whole training set, not single data point
    
    int train_amount = split_size*data_amount;
    int test_amount = data_amount-train_amount;
    
    hd_model->X_train = X_train;
    fit(X_train, in_dim, train_amount);
    hd_model->X_test = X_test;
    fit(X_test, in_dim, test_amount);
    hd_model->y_train = y_train;
    hd_model->y_test = y_test;
    
    hd_model->in_dim = in_dim;
    hd_model->out_dim = out_dim;
    hd_model->lr = lr;
    hd_model->split_size = split_size;

    hd_model->class_size = 12;

    lrp* linear_rp;
    init_lrp(linear_rp, in_dim, out_dim);
    hd_model->encoder = linear_rp;
    
    //class_hvs = 12 * out_dim 
    hd_model->class_hvs = (char**) malloc(sizeof(char)*12); //hard coding the number of classes
    for(int i = 0; i < 12; i ++){ //hard coding the number of classes
        hd_model->class_hvs[i] = (char*) malloc(sizeof(char)*out_dim);
        for(int j = 0; j < out_dim; j ++){
            hd_model->class_hvs[i][j] = 0;
        }
    }

    hd_model->train_encs = (char**) malloc(sizeof(char)*train_amount);
    for(int k = 0; k < out_dim; k ++){
        hd_model->train_encs[k] = (char*) malloc(sizeof(char)*out_dim);
    }

}

void train(hdModel* model){
    int n_train = (model->split_size)*(model->data_amount);
    for(int i = 0; i < n_train; i ++){
        float* sample = (model->X_train)[i];
        char label = model->y_train[i];
        char* enc = encode(model->encoder, sample);
        float* cos_similarity = (float*) malloc(sizeof(float)*(model->class_size)); //hard coding with number of classes
        //cosine similairty: enc(d*1) * (n*d) --> (1*d)* (d*n) ---> (n*1) ??
        cosine_similarity(cos_similarity, model, enc);
        char index_pred = max_index(cos_similarity, model->class_size);
        if(index_pred != label){
            add_enc(model->class_hvs[label], enc, model->out_dim);
            subtract_enc(model->class_hvs[index_pred], enc, model->out_dim);
        }
        append_enc(model->train_encs, enc, i, model->out_dim);
    }
}

void test(hdModel* hd_model){
    int n_test = (hd_model->data_amount)*((1-hd_model->split_size)*hd_model->data_amount);
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
        printf(count);
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

void fit(float** input_feature, int in_dim, int data_amount){
    for(int i = 0; i < data_amount; i ++){
        float* feature = input_feature[i];
        float mean = calculate_mean(feature, in_dim);
        float standard_deviation = calculate_standard_deviation(feature, in_dim);
        for(int j = 0; j < in_dim; j ++){
            feature[j] = (feature[j]-mean)/standard_deviation;
        }
    }
}



float calculate_mean(float* arr, int size){
    float sum = 0.0;
    for(int i = 0; i < size; i ++){
        sum =+ arr[i];
    }
    return sum/size;
}

float calculate_standard_deviation(float* arr, int size){
    float mean = calculate_mean(arr, size);
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += pow(arr[i] - mean, 2);
    }
    return sqrt(sum / (size - 1)); 
}

float dot_product(float* vec1, float* vec2, int length){
    //cosine similairty: enc(d*1) * (n*d) --> (1*d)* (d*n) ---> (n*1)
    float dot = 0.0; 
    for(int i = 0; i < length; i ++){
        dot += vec1[i] * vec2[i];
    }
    return dot;
}

float magnitude(float* vec1, int length){
    float magnitude = 0.0;
    for(int i = 0; i < length; i ++){
        magnitude += vec1[i]*vec1[i];
    }
    return sqrt(magnitude);
}


void cosine_similarity(float* cosine_similarity, hdModel* model, float* enc){
    for (int i = 0; i < model->class_size; i ++){
        float similarity = 0.0;
        float dot = dot_product(enc, model->class_hvs[i], model->out_dim);
        float magnitude_enc = magnitude(enc, model->out_dim);
        float magnitude_class_vec = magnitude(model->class_hvs[i], model->out_dim);
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

void add_enc(float* vec1, char* enc, int length){
    for(int i = 0; i < length; i ++){
        vec1[i] = vec1[i] + enc[i];
    }
}

void subtract_enc(float* vec1, char* enc, int length){
    for(int i = 0; i < length; i ++){
        vec1[i] = vec1[i] - enc[i];
    }
}

void append_enc(char** total_train_encs, char* enc, int index, int length){
    for(int i = 0; i < length; i ++){
        total_train_encs[index][i] = enc[i];
    }
}