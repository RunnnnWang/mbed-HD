#include <hd_model.h>


void init_hd_model(hdModel* hd_model, float** all_data, int* all_label){ //X_test and y_test contains the whole training set, not single data point
    
//closed shuffle
  shuffle(all_data, all_label, DATA_SIZE, 6); //arbitrary random state 42  


    // float printMean = 0;
    // float printSum = 0;
    // float printStd = 0;
    // float printyMean = 0;
    // float printySum = 0;
    // float printyStd = 0;

    // for(int i = 0; i < TRAIN_AMOUNT; i ++){
    for(int i = 0; i < DATA_IN_DIM; i ++){
        //mean, sum, std initialization
        float mean = 0.0;
        float sum = 0.0;
        float std = 0;
        
        //mean calculation
        // for(int j = 0; j < DATA_IN_DIM; j ++){
        for(int j = 0; j < TRAIN_AMOUNT; j ++){
            mean += all_data[j][i];  
            
        }
        mean = mean/TRAIN_AMOUNT;
        
        //std calculation
        for(int j = 0; j < TRAIN_AMOUNT; j++){
            sum += pow(all_data[j][i] - mean, 2);
            // printStd += pow(all_data[i][j] - mean, 2);

        }
        std = sqrt(sum /(TRAIN_AMOUNT)); 

        //fit on the curr data
        for(int j = 0; j < TRAIN_AMOUNT; j ++){
            hd_model->X_train[j][i] = (all_data[j][i]-mean)/std;
            // printSum += (all_data[i][j]-mean)/std;

        }

        for(int j = 0; j < TEST_AMOUNT; j ++){
            hd_model->X_test[j][i] = (all_data[TRAIN_AMOUNT+j][i]-mean)/std;
        }
        // printMean += mean;
        // printSum += sum;
        // printStd = std;

        //initialize the y train
        // hd_model->y_train[i] = (char)all_label[i];
        // printySum += (int)all_label[i];
        // printyStd += ((float)all_label[i] - 5.537) * ((int)all_label[i] - 5.537);

    }
    // printStd = sqrt(printStd / TRAIN_AMOUNT / DATA_IN_DIM);
    // printMean = printSum / TRAIN_AMOUNT;
    // printyMean = printySum / TRAIN_AMOUNT;
    // printyStd = sqrt(printyStd / TRAIN_AMOUNT);
    // printf("mean:%f, sum:%f, std:%f\n", printMean, printSum, printStd);
    // printf("ymean:%f, ysum:%f, ystd:%f\n", printyMean, printySum, printyStd);

    // printMean = 0.0;
    // printSum = 0.0;
    // printStd = 0.0;
    // printyMean = 0.0;
    // printySum = 0.0;
    // printyStd = 0.0;


    // int index = 0; 
    // for(int i = TRAIN_AMOUNT; i < DATA_SIZE; i ++){
    // printStd = sqrt(printStd / (DATA_SIZE - TRAIN_AMOUNT) / DATA_IN_DIM);
    // printMean = printMean / (DATA_SIZE - TRAIN_AMOUNT);
    // printyMean = printySum / (DATA_SIZE - TRAIN_AMOUNT);
    // printyStd = sqrt(printyStd / (DATA_SIZE - TRAIN_AMOUNT));
    // printf("mean:%f, sum:%f, std:%f\n", printMean, printSum, printStd);
    // printf("ymean:%f, ysum:%f, ystd:%f\n", printyMean, printySum, printyStd);

    for (int i = 0; i < TRAIN_AMOUNT; i++){
        hd_model->y_train[i] = (char)all_label[i];
    }

    for (int i = TRAIN_AMOUNT; i < DATA_SIZE; i++){
        hd_model->y_test[i - TRAIN_AMOUNT] = (char)all_label[i];
    }


    //initialize class hvs
    for(int i = 0; i < 12; i ++)
        for(int j = 0; j < DATA_OUT_DIM; j ++){
            hd_model->class_hvs[i][j] = 0;
        }


    //initialize linear random projection
    init_lrp(hd_model);
}

    




void train(hdModel* model){
    int initial_class_hv[12] = {0};
    for(int i = 0; i < TRAIN_AMOUNT; i ++){
        encode(model, i);
        char label = model->y_train[i];
        if(!initial_class_hv[label]){
            for(int j = 0; j < DATA_OUT_DIM; j ++){
                model->class_hvs[label][j] = model->train_encs[i][j];
            }
            initial_class_hv[label] = 1;
            continue;
        }
        //calculate cosine similarity
        int index_pred = 0;
        float biggest_similarity = -10000;
        for(int j = 0; j < CLASS_AMOUNT; j ++){
            if(!initial_class_hv[j]){
                continue;
            }
            float curr_similarity = 0;
            //dot product of the class hv and encoding, magnitude of the class hv and encoding
            int dot = 0; 
            float hv_magnitude = 0;
            float encoding_magnitude = 0;
            for(int k = 0; k < DATA_OUT_DIM; k ++){
                dot += model->class_hvs[j][k] * model->train_encs[i][k];
                hv_magnitude += model->class_hvs[j][k]*model->class_hvs[j][k];
                encoding_magnitude += model->train_encs[i][k]*model->train_encs[i][k];
            }
            curr_similarity = dot/(sqrt(hv_magnitude)*sqrt(encoding_magnitude));
            if(curr_similarity > biggest_similarity){
                index_pred = j;
                biggest_similarity = curr_similarity;
            }
        }

        if(index_pred != label){
            //add encoding + subtract encoding + append to 
            for(int j = 0; j < DATA_OUT_DIM; j ++){
                model->class_hvs[label][j] +=  model->train_encs[i][j];
                model->class_hvs[index_pred][j] -= model->train_encs[i][j];
            }
        }
    }
}

void test(hdModel* model){
    int correct_count = 0; 
    for(int i = 0; i < TEST_AMOUNT; i++){
        
        float curr_enc[DATA_OUT_DIM] = {0};
        //encode
        for (int j = 0; j < DATA_OUT_DIM; j ++) {
            for (int k = 0; k< DATA_IN_DIM; k ++) { 
                curr_enc[j] += model->projection[j][k]*model->X_test[i][k];
        }
            curr_enc[j] = sign(curr_enc[j]);
        }

        //calculate cosine similarity
        int index_pred = 0;
        float biggest_similarity = -10000;
        for(int j = 0; j < CLASS_AMOUNT; j ++){
            float curr_similarity = 0.0;
            //dot product of the class hv and encoding, magnitude of the class hv and encoding
            int dot = 0; 
            int hv_magnitude = 0;
            int encoding_magnitude = 0;
            for(int k = 0; k < DATA_OUT_DIM; k ++){
                dot += model->class_hvs[j][k]*curr_enc[k];
                hv_magnitude += model->class_hvs[j][k]*model->class_hvs[j][k];
                encoding_magnitude += curr_enc[k]*curr_enc[k];
            }
            curr_similarity = dot/(sqrt(hv_magnitude)*sqrt(encoding_magnitude));
            if(curr_similarity > biggest_similarity){
                index_pred = j;
                biggest_similarity = curr_similarity;
            }
        }

        char label = model->y_test[i];
        if(index_pred == label){
            correct_count += 1;
        }    
    }
    float score = (float)correct_count/(float)TEST_AMOUNT/1.0;
    
    printf("accuracy: %f \n", score);
    
}


        

void retrain(hdModel* model){
    int count = 0; 
    for(int e = 0; e < 4; e++){
        count = 0;
        for(int i = 0; i < TRAIN_AMOUNT; i ++){
            
            //get encoding + label
            char curr_enc[DATA_OUT_DIM];
            for(int j = 0; j < DATA_OUT_DIM; j ++){
                curr_enc[j] = model->train_encs[i][j];
            }
            char label = model->y_train[i];

            //initialize pred and similairty
            int index_pred = 0;
            float biggest_similarity = -10000;
            
            for(int j = 0; j < CLASS_AMOUNT; j ++){
                float curr_similarity = 0.0;
            //dot product of the class hv and encoding, magnitude of the class hv and encoding
                int dot = 0; 
                int hv_magnitude = 0;
                int encoding_magnitude = 0;
                for(int k = 0; k < DATA_OUT_DIM; k ++){
                    dot += model->class_hvs[j][k]*curr_enc[k];
                    hv_magnitude += model->class_hvs[j][k]*model->class_hvs[j][k];
                    encoding_magnitude += curr_enc[k]*curr_enc[k];
                }
                curr_similarity = dot/(sqrt(hv_magnitude)*sqrt(encoding_magnitude));
                if(curr_similarity > biggest_similarity){
                    index_pred = j;
                    biggest_similarity = curr_similarity;
                }
            }
            if(index_pred != label){
            //add encoding + subtract encoding + append to 
                for(int j = 0; j < DATA_OUT_DIM; j ++){
                    model->class_hvs[label][j] +=  curr_enc[j];
                    model->class_hvs[index_pred][j] -= curr_enc[j];
                }
                count += 1;
            }
        }
        printf("count %d", count);
    }
}



//---------------------helper method-------------------------------------




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





//----------------------------------------linear random porjection methods----------------------------
void init_lrp(hdModel* model){
    //init projection D*n
    for (int i = 0; i < DATA_OUT_DIM; i ++) {
        for (int j = 0; j < DATA_IN_DIM; j ++) {
            model->projection[i][j] = sign(generate_normal_random_float());
        }
    }
}

void encode(hdModel* model, int index){ //x.shape = n * 1
    for (int i = 0; i < DATA_OUT_DIM; i ++) {
        for (int j = 0; j < DATA_IN_DIM; j ++) { 
            model->train_encs[index][i] += model->projection[i][j]*model->X_train[index][j];
        }

        model->train_encs[index][i] = sign(model->train_encs[index][i]);
    }
}

//-----------------helper method------------------------------------------------
float generate_normal_random_float() {
    float u1, u2;
    float z0;  // Only using z0, but the Box-Muller produces two values z0 and z1

    // Generate two uniform random numbers between 0 and 1
    u1 = (float)rand() / RAND_MAX;
    u2 = (float)rand() / RAND_MAX;

    // Apply the Box-Muller formula
    z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);

    return z0;
}

char sign(float num){
    if (num > 0) {
        return 1;
    }
    if (num < 0) {
        return -1;
    }  
    return 0;
}

