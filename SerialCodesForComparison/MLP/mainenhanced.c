#include <stdio.h>
#include<string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include "Matrix.h"
#include "MLP.h"
#include "MNIST_for_C-master/mnist.h"

#define EMPTY printf("\n\n");


#define train_count 1100
#define test_count 1000
int main()
{



   /* float * img_ = malloc(sizeof(float)*train_count*SIZE);
    for(int h = 0; h <train_count ; h++)
        for(int w = 0; w<SIZE; w++)
    {
        img_[h*SIZE+w] = train_image[h][w];
    }
*/
    Vector2D * image_input = CreateVector2D(train_image, train_count, SIZE);
    Vector2D * data_set = ColumnVectorAdd(image_input, 1.0);
    //DestroyVector2D(image_input);



    /*img_ = malloc(sizeof(float)*train_count);
    for(int index = 0; index < train_count; index++)
    {
        img_[index] = train_label[index];
    }
    */
    Vector2D * image_labels = CreateVector2D(train_label, train_count, 1 );
    Vector2D * one_hot_labels = CreateOneHot(image_labels, 10);
    //DestroyVector2D(image_labels);



    /*img_ = malloc(sizeof(float)*test_count*SIZE);
    for(int h = 0; h <test_count ; h++)
        for(int w = 0; w<SIZE; w++)
    {
        img_[h*SIZE+w] = test_image[h][w];
    }*/

    Vector2D * test = CreateVector2D(test_image, test_count, SIZE);
    Vector2D * test_image = ColumnVectorAdd(test, 1.0);
    //DestroyVector2D(test);

    /*img_ = malloc(sizeof(float)*test_count);
    for(int i=0; i < test_count; i++)
    {
        img_[i] = test_label[i];
    }*/

    Vector2D * test_label_indices = CreateVector2D(test_label, test_count, 1);
    Vector2D * test_labels = CreateOneHot(test_label_indices, 10);
    //DestroyVector2D(test_label_indices);


    CreateMLP(4, 784, 100, 20, 10);
    printf("\n\nMLP has been created...\n\n");
    Vector2D * output, * log_output, * scalar_result, * pairwise ;
    Vector2D * first_vector = VectorSlice2D(data_set, 0, 1, 0, data_set->width);
    Vector2D * second_vector = VectorSlice2D(data_set, 800, 801, 0, data_set->width);

    for(int iteration = 0 ; iteration < 100; iteration++)
    {
        output = FeedForward(data_set);

        log_output = Log2D(output);
        Vector2D * one_hot_labels_copy = CopyVector(one_hot_labels);

        pairwise = MatrixPairwiseProduct(one_hot_labels_copy, log_output);
        DestroyVector2D(log_output);

        printf("\n\n***********************Iteration : %d - Error : %f\n", iteration, -Sum2D(pairwise));
        DestroyVector2D(pairwise);
       // printf("\nBackpropagation\n");
        BackPropagate(data_set, one_hot_labels);




    }
    printf( "Tamamdır...\n\n");
    Vector2D * final_output = FeedForward(test_image);
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

    return 0;
}
