#include <stdio.h>
#include<string.h>
#include <stdlib.h>
#include <stdarg.h>
#include<time.h>
#include <math.h>
#include "Matrix.h"
#include "MLPClassifier.h"
// https://github.com/takafumihoriuchi/MNIST_for_C
#include "MNIST_for_C-master/mnist.h"

#define EMPTY printf("\n\n");


#define train_count 1200
#define test_count 5000
int main()
{

    clock_t begin_time = clock();
    double min_error = 10000000;
    load_mnist();

    printf("SIZE : %d", SIZE);

    float * train_data = malloc(sizeof(float)*train_count*784);
    for(int h = 0; h < train_count; h++)
    {
        for(int w=0; w < 784; w++)
        {
            train_data[h*784+w] = train_image[h][w];
        }

    }

    float * train_label_float = malloc(sizeof(float)*train_count);
     for(int h = 0; h < train_count; h++)
    {
            train_label_float[h] = train_label[h];
    }

    float * test_data = malloc(sizeof(float)*test_count*784);
     for(int h = 0; h < test_count; h++)
    {
        for(int w=0; w < 784; w++)
        {
            test_data[h*784+w] = test_image[h][w];
        }

    }

     float * test_label_float = malloc(sizeof(float)*test_count);
     for(int h = 0; h < test_count; h++)
    {
            test_label_float[h] = test_label[h];
    }

    Vector2D * data_set = CreateVector2D(train_data, train_count, SIZE);

    Vector2D * image_labels = CreateVector2D(train_label_float, train_count, 1 );
    DisplayVector2D(image_labels);



    Vector2D * one_hot_labels = CreateOneHot(image_labels, 10);


    Vector2D * test_image_fin = CreateVector2D(test_data, test_count, SIZE);

    Vector2D * test_label_indices = CreateVector2D(test_label_float, test_count, 1);
    Vector2D * test_labels = CreateOneHot(test_label_indices, 10);



    CreateMLP(0.0001, 3, 784, 100, 10);
    printf("\n\nMLP has been created...\n\n");
    Vector2D * output, * log_output, * scalar_result, * pairwise ;

    float total_error;

    for(int iteration = 0 ; iteration < 50; iteration++)
    {
        total_error = MLPClassifierError(data_set, one_hot_labels);
        printf("\nIteration %d - Error : %f\n", iteration, total_error);
       if(min_error > total_error)
    {
        min_error = total_error;

    for(int weight_index=0; weight_index < mlp_layer_count - 1;weight_index++)
    {
        DestroyVector2D(best_weight_array[weight_index]);
        DestroyVector2D(best_bias_array[weight_index]);

        best_weight_array[weight_index] = CopyVector(weight_array[weight_index]);
        best_bias_array[weight_index] = CopyVector(bias_array[weight_index]);
    }

    }

        // printf("\nBackpropagation\n");
        BackPropagate(data_set, one_hot_labels);




    }
     printf("\n\nMLP will be configured to best value with error of %f\n\n", min_error);
     for(int weight_index=0; weight_index < mlp_layer_count - 1;weight_index++)
    {

        weight_array[weight_index] = CopyVector(best_weight_array[weight_index]);
        bias_array[weight_index] = CopyVector(best_bias_array[weight_index]);
    }


    Vector2D * final_output = FeedForward(test_image_fin);
    Vector2D* nn_indices = ArgMax2D(final_output);
    Vector2D * real_indices = ArgMax2D(one_hot_labels);
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
