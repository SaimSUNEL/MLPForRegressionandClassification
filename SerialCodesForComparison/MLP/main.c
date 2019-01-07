#include <stdio.h>
#include<string.h>
#include <stdlib.h>
#include <stdarg.h>
#include<time.h>
#include <math.h>
#include "Matrix.h"
#include "MLPClassifier.h"
#include "DataLoad.h"
// https://github.com/takafumihoriuchi/MNIST_for_C
#include "MNIST_for_C-master/mnist.h"

#define EMPTY printf("\n\n");


#define train_count 1200
#define test_count 5000
int main()
{
    clock_t error_calculation_start, error_calculation_end;
    clock_t train_start, train_end;

    clock_t begin_time = clock();
    double min_error = 10000000;

    Vector2D * data_set = load_text_data();
	Vector2D * labels_ = load_label_data();
	printf("\nData loaded...\n");
    Vector2D * one_hot_labels = CreateOneHot(labels_, 4);

    Vector2D * test_data = load_test_text_data();
    Vector2D * lab = load_test_label_data();
    Vector2D *one_hot_test = CreateOneHot(lab, 4);




    CreateMLP(0.0001, 3, 32754, 128, 4);
    printf("\n\nMLP has been created...\n\n");
    Vector2D * output, * log_output, * scalar_result, * pairwise ;

    #define BATCH_SIZE 32
    float total_error;

    for(int iteration = 0 ; iteration < 50; iteration++)
    {

        total_error = 0.0;
        error_calculation_start = clock();

        train_start = clock();
    for(int batch_index = 0; batch_index <50;batch_index++)///< data_set->height/BATCH_SIZE; batch_index++)
        {

            printf("\nbatch_index : %d\n", batch_index);
             Vector2D * batch_set = CreateVector2D(data_set->data+(batch_index)*BATCH_SIZE*data_set->width, BATCH_SIZE, data_set->width);
        Vector2D * batch_label = CreateVector2D(one_hot_labels->data+batch_index*BATCH_SIZE*one_hot_labels->width, BATCH_SIZE, one_hot_labels->width);

        // printf("\nBackpropagation\n");
        BackPropagate(batch_set, batch_label);

        total_error += MLPClassifierError(batch_set, batch_label);


        }
        error_calculation_end = clock();
        printf("\nIteration %d - whole data error calculation time : %f\n",iteration, (double)(error_calculation_end - error_calculation_start) / CLOCKS_PER_SEC);
        printf("\nIteration %d - Error : %f\n", iteration, total_error);



       if(min_error > total_error)
    {
        min_error = total_error;

    for(int weight_index=0; weight_index < mlp_layer_count - 1;weight_index++)
    {
        //DestroyVector2D(best_weight_array[weight_index]);
        //DestroyVector2D(best_bias_array[weight_index]);

        best_weight_array[weight_index] = CopyVector(weight_array[weight_index]);
        best_bias_array[weight_index] = CopyVector(bias_array[weight_index]);
    }

    }



    }
     printf("\n\nMLP will be configured to best value with error of %f\n\n", min_error);
     for(int weight_index=0; weight_index < mlp_layer_count - 1;weight_index++)
    {

        weight_array[weight_index] = CopyVector(best_weight_array[weight_index]);
        bias_array[weight_index] = CopyVector(best_bias_array[weight_index]);
    }


    Vector2D * final_output = FeedForward(test_data);
    Vector2D* nn_indices = ArgMax2D(final_output);
    Vector2D * real_indices = ArgMax2D(one_hot_test);
    int correct_count=0, false_count = 0;
    for(int h = 0 ; h<nn_indices->height; h++)
    {
        if(abs(nn_indices->data[h]-real_indices->data[h])<0.00000001)
        {
            correct_count++;
        }
        else false_count++;
    }
    printf("\n\nAccuracy : %f\n", ((float)correct_count/(correct_count+false_count)*100.0));

    //printf("\n\nTrue labels \n\n");
   //DisplayVector2D(test_labels);

 //   exit(0);
/*
    float y[] = {0, 2};
    Vector2D * y_label = CreateVector2D(y, 2, 1);
    Vector2D * y_one_hot = CreateOneHot(y_label, 3);

    float data [] = {+1, 1, 2, 3,
                     +1, 4, 5, 6};
    Vector2D * input_vector = CreateVector2D(data, 2, 4);


    CreateMLP(4, 3, 10, 5,  3);
    printf("First w1 : \n");
    DisplayVector2D(weight_array[0]);
    EMPTY;
    printf("First w2 : \n");
    DisplayVector2D(weight_array[1]);
    EMPTY;

    DisplayVector2D(FeedForward(input_vector));
    EMPTY;


    printf("First layer res : \n");
    DisplayVector2D(layer_results[0]);
    EMPTY;EMPTY;
    printf("Second layer res : \n");
    DisplayVector2D(layer_results[1]);

    EMPTY;

    EMPTY;EMPTY;
    printf("last layer\n");
    DisplayVector2D(layer_results[2]);
    EMPTY;EMPTY;
    BackPropagate(input_vector, y_one_hot);

*/
/*srand(62);
    float d1 [] = {1, 2, 3, 0.1,
                    0.0, 0.0, 0.0, 0.0};
    Vector2D * v1 = CreateVector2D(d1, 2, 4);

    float d2 [] ={1, 0.03, 0.04, 0.2,
                  0.7, 0.05, 0.06, 0.1};

    Vector2D * v2 = CreateVector2D(d2, 2, 4);
    CreateMLP(4, 3, 5,4,6);
    printf("first w0 : \n");
    DisplayVector2D(weight_array[0]);
    printf("\nw1 : \n");
    DisplayVector2D(weight_array[1]);
    printf("\nW3 : \n");
    DisplayVector2D(weight_array[2]);

    EMPTY;

    DisplayVector2D(FeedForward(v1));
    EMPTY;
    //DisplayVector2D(FeedForward(v2));
*/

   /* int rakam [10] = {0};
    load_mnist();
    int i = 0;
    for( ; i < 1100;i++)
    {
        rakam[train_label[i]]++ ;
    }

    for(i=0;i < 10;i++)
    {
        printf("%d - %d\n", i, rakam[i]);
    }



exit(0);*/



clock_t end_time = clock();
double time_spent = (double)(end_time - begin_time) / CLOCKS_PER_SEC;
printf("\n\nElapsed exact time %f", time_spent);

    return 0;
}
