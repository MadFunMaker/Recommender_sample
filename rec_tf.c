#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

#define nItem 150000
#define maxNnz 47000000
#define nTrashStr 100

int main()
{

	//Time measurement
	clock_t begin;
	clock_t end;
	begin = clock();

	// Initialize variables
    FILE *meta_fp;
    int *dic;
    int item, i;
    char trashStr[nTrashStr];
    char contextStr[nTrashStr];

    // Read meta data to build dictionary in order to change matrix to tensor.
    // For now, context is Artist ID
    dic = (int*) malloc(sizeof(int)*nItem);
    meta_fp = fopen("full_meta_data.txt", "r");
    while (fscanf(meta_fp, "%d\t%s\t%s", &item, trashStr, contextStr) >= 0) {
        int contextLen = strlen(contextStr);
        for (i=0; i<contextLen; ++i) if (!isdigit(contextStr[i])) break;
        if (i==contextLen) dic[item] = atoi(contextStr);
//    	printf("%d\t%s\n", item, contextStr);
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
//    	printf("%d\t%d\t%f\n", users[nNnz], items[nNnz], ratings[nNnz]);
        ++nNnz;
    }
    fclose(rating_fp);

    // Time Measurement to read data
    end = clock();
    printf("[TS for RS] %f sec spent to read data\n", (double)(end-begin)/CLOCKS_PER_SEC);
    begin = clock();
    
    
    
    end = clock();
    printf("[TS for RS] %f sec spent to factorize data\n", (double)(end-begin)/CLOCKS_PER_SEC);
    return 0;
}
