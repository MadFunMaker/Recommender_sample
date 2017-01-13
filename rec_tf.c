#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

#define nUser 420000
#define nContext 1500000
#define nItem 150000
#define maxNnz 47000000
#define nTrashStr 1000
#define nRank 100
#define alpha 0.0002
#define lambda 0.02
#define epoch 10000

float randomVal()
{
    return (float)rand() / (float)RAND_MAX;
}

int main()
{

	//Time measurement
	clock_t begin;
	clock_t end;
	begin = clock();

	// Initialize variables
    FILE *meta_fp;
    int *item_context_dic;
    int item, i;
    char trashStr[nTrashStr];
    char contextStr[nTrashStr];

    // Read meta data to build dictionary in order to change matrix to tensor.
    // For now, context is Artist ID
    item_context_dic = (int*) malloc(sizeof(int)*nItem);
    meta_fp = fopen("full_meta_data.txt", "r");
    int cnt =0;
    while (fscanf(meta_fp, "%d\t%s\t%s", &item, trashStr, contextStr) >= 0) {
        int contextLen = strlen(contextStr);
        for (i=0; i<contextLen; ++i) if (!isdigit(contextStr[i])) break;
        if (i==contextLen) {
            item_context_dic[item] = atoi(contextStr);
            if (cnt%10000==0) printf("%d\t%s\n", item, contextStr);
            ++cnt;
        }
    }
    fclose(meta_fp);
    printf("[TF for RS] Item-Context dictionary stored in memory\n");
    
    FILE *rating_fp;
    int *users, *items;
    float *ratings;

    // Read rating matrix data
    rating_fp = fopen("train_data", "r");
    users = (int*) malloc(sizeof(int)*maxNnz);
    items = (int*) malloc(sizeof(int)*maxNnz);
    ratings = (float*) malloc(sizeof(float)*maxNnz);
    int nNnz = 0;
    while (fscanf(rating_fp, "%d\t%d\t%f", &users[nNnz], &items[nNnz], &ratings[nNnz]) >= 0) {
    	// printf("%d\t%d\t%f\n", users[nNnz], items[nNnz], ratings[nNnz]);
        if (ratings[nNnz] > 0) ++nNnz;
    }
    fclose(rating_fp);

    // Time Measurement to read data
    end = clock();
    printf("[TS for RS] %f sec spent to read data\n", (double)(end-begin)/CLOCKS_PER_SEC);
    begin = clock();

    
    // Regularized CP Tensor factorization with SGD optimization
    float **A, **B, **C;
    int j;
    A = (float**) malloc(sizeof(float*)*nUser);
    B = (float**) malloc(sizeof(float*)*nItem);
    C = (float**) malloc(sizeof(float*)*nContext);
    for (i=0; i<nUser; ++i) A[i] = (float*)malloc(sizeof(float)*nRank);
    for (i=0; i<nItem; ++i) B[i] = (float*)malloc(sizeof(float)*nRank);
    for (i=0; i<nContext; ++i) C[i] = (float*)malloc(sizeof(float)*nRank);        

    srand((unsigned)time(NULL));
    for (i=0; i<nUser; ++i) for (j=0; j<nRank; ++j) A[i][j] = randomVal();
    for (i=0; i<nItem; ++i) for (j=0; j<nRank; ++j) B[i][j] = randomVal();
    for (i=0; i<nContext; ++i) for (j=0; j<nRank; ++j) C[i][j] = randomVal();

    int step;
    int steps = epoch;
    int k,r,idx;
    float val, rmse_error;

    end = clock();
    printf("[TS for RS] %f sec spent to initialize factor matrices\n", (double)(end-begin)/CLOCKS_PER_SEC);
    begin = clock();


    // Check for optimization algoirhtm to converge
    int history_size = 1000;
    float rmse_history[history_size];
    int check_history_flag = 0;
    float epsilon_converge = 0.0000001;
    float target_error = 0.3;

    for (step=0; step<steps; ++step) {
        for (idx=0; idx<nNnz; ++idx) {
            i = users[idx];
            j = items[idx];
            k = item_context_dic[items[idx]];
            val = ratings[idx];

            if (val > 0) {
                float predicted_val = 0;
                for (r=0; r<nRank; ++r) predicted_val += A[i][r]*B[j][r]*C[k][r];
                float eijk = val - predicted_val;
                for (r=0; r<nRank; ++r) {
                    A[i][r] = A[i][r] + alpha * (2 * eijk * B[j][r] * C[k][r] - lambda * A[i][r]);
                    B[j][r] = B[j][r] + alpha * (2 * eijk * A[i][r] * C[k][r] - lambda * B[j][r]);
                    C[k][r] = C[k][r] + alpha * (2 * eijk * B[j][r] * A[i][r] - lambda * C[k][r]);
                }
            }
        }
        
        rmse_error = 0;
        for (idx=0; idx<nNnz; ++idx) {
            i = users[idx];
            j = items[idx];
            k = item_context_dic[items[idx]];
            val = ratings[idx];

            if (val > 0) {
                float predicted_val = 0;
                for (r=0; r<nRank; ++r) predicted_val += A[i][r]*B[j][r]*C[k][r];
                rmse_error += pow(val-predicted_val,2);
            }
        }
        rmse_error = sqrt(rmse_error/nNnz);
        if (rmse_error<target_error) {
            printf("[TF for RS] Target RMSE(%f) is reached.\n", target_error);
            break;
        } 

        // Convergence test
        rmse_history[step%history_size] = rmse_error;
        if (step%history_size == history_size-1) check_history_flag = 1;
        if (check_history_flag) {
            for (r=0; r<history_size; ++r) if (fabsf(rmse_history[r]-rmse_error) > epsilon_converge) break;
            if (r==history_size) {
                printf("[TF for RS] Optimization algorithm converges.\n");
                break;
            }
        }
        if (step%1000==0) printf("[TF for RS] %d/%d steps done.\n",step,steps);
    }

    end = clock();
    printf("[TS for RS] %f sec spent to factorize data\n", (double)(end-begin)/CLOCKS_PER_SEC);
    printf("[TF for RS] RMSE : %f\n", rmse_error);

    return 0;
}
