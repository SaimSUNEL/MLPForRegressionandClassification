#include <stdio.h>
#include<string.h>
#include <stdlib.h>
#include <stdarg.h>
#include<time.h>
#include <math.h>
#include "Matrix.h"
#include <pthread.h>
#include "MLPClassifier.h"
// https://github.com/takafumihoriuchi/MNIST_for_C
#include "MNIST_for_C-master/mnist.h"
#include "DataLoad.h"
#define EMPTY printf("\n\n");
#define train_count 1600
#define test_count 1000



// Each thread will be responsible for some part of whole training data...
// Each thread performs forward propagation and backward propagation with the data it is responsible for...
struct MyThread {
// Thread's assigned trained data...
Vector2D * data_set ;
// Training data's corresponding ground truth...
Vector2D * one_hot_labels;
// Each thread is differentiated by a number...
int thread_id ;

// Error value on the given training data part...
float error;

} ;


// Simple all threads performs forward propagation, and then backward propagation...
// Each threads' forward propagation results are stored in global array with thread id's...
// Each thread has its own location to put its result data to be used in back propagation later...

void * thread_function( void * ptr)
{
  struct MyThread * current_thread = (struct MyThread *)ptr;

    // Measure error with thread specific data...
    current_thread->error = MLPClassifierError(current_thread->data_set, current_thread->one_hot_labels, current_thread->thread_id);
       // printf("\nBackpropagation\n");

    // Apply back propagation and store the weight changes in a global location...
    // Each thread performs this operation, when all finishes the changes are accumulated and applied to
    // our neural network....
    BackPropagate(current_thread->data_set, current_thread->one_hot_labels, current_thread->thread_id);




}


#define THREAD_COUNT 8

int main()
{
    double min_error = 100000;
    printf("Program started");

    clock_t error_calculation_start, error_calculation_end;
    clock_t train_start, train_end;

    // To measure the execution time of the program...
    clock_t begin_time = clock();
     Vector2D * data_set = load_text_data();
	Vector2D * labels_ = load_label_data();
	printf("\nData loaded...\n");
    Vector2D * one_hot_labels = CreateOneHot(labels_, 4);

    Vector2D * test_data = load_test_text_data();
    Vector2D * lab = load_test_label_data();
    Vector2D *one_hot_test = CreateOneHot(lab, 4);

    printf("\nTest : \n");
    DisplayVector2D(one_hot_test);




    CreateMLP(0.0001, THREAD_COUNT, 3, 32754, 128, 4);
    printf("\n\nMLP has been created...\n\n");
    Vector2D * output, * log_output, * scalar_result, * pairwise ;;



    //Vector2D Thread_Dataset[THREAD_COUNT];
    //Vector2D Thread_Onehotlabes[THREAD_COUNT];
    struct MyThread threads [THREAD_COUNT];
    pthread_t thread_ids[THREAD_COUNT];

    //We are partitioning the dataset for each thread...
    for(int slice_index = 0 ; slice_index<THREAD_COUNT;slice_index++ )
    {
        threads[slice_index].data_set = VectorSlice2D(data_set, slice_index*(train_count/THREAD_COUNT), (slice_index+1)*(train_count/THREAD_COUNT),0, data_set->width );
        threads[slice_index].one_hot_labels = VectorSlice2D(one_hot_labels, slice_index*(train_count/THREAD_COUNT), (slice_index+1)*(train_count/THREAD_COUNT), 0, one_hot_labels->width);
        threads[slice_index].thread_id = slice_index;

    }

    // Here we are applying batch learning, we are using whole data set to update the weights....
    // Each thread is responsible for some part of the data
    // Parallelly each thread calculates the weight updates for the data given to it...
    // When all threads finish their job, we are applying changes to network...

    // For each iterataion we are creating a group of threads to process data...
    float total_error = 0.0;

    // We are runnig over our test data many times while training...
    for(int iteration = 0 ; iteration < 10; iteration++)
    {
        train_start = clock();
        struct timespec begin, end;
        clock_gettime(CLOCK_MONOTONIC_RAW, &begin);
        // We are creating thread each of which is responsible for some part of the whole data...
        for(int thread = 0; thread< THREAD_COUNT; thread++)
        {

            if(pthread_create(&thread_ids[thread], NULL, thread_function, &threads[thread]))
                {

                fprintf(stderr, "Error creating thread\n");
                return 1;

                }

        }

        // We are waiting all threads to finish their execution, the weight update must be done synchronously..
        for(int thread = 0; thread<THREAD_COUNT; thread++)
        {
                     if(pthread_join(thread_ids[thread], NULL)) {

                        fprintf(stderr, "Error joining thread\n");
                        return 2;

                        }

        }


    // To calculate train error we are collecting each threads' individual error....
    total_error = 0.0;
    for(int thread = 0; thread < THREAD_COUNT; thread++)
    {
        total_error += threads[thread].error;

    //We are updating our neural nets weights by accumulating all related changes from threads...

    for(int weight_index=0; weight_index < mlp_layer_count - 1;weight_index++)
    {

        MatrixAdd(weight_array[weight_index], ScalarMatrixProduct(learning_rate, weight_update_container[thread][weight_index]));
        MatrixAdd(bias_array[weight_index], ScalarMatrixProduct(learning_rate, bias_update_container[thread][weight_index]));
    }

    // we are destroying the layer results weight updates because they will be used again and wasts memory..
    for(int i = 0; i < mlp_layer_count-2; i++)
    {
       DestroyVector2D(layer_results[thread][i]);
       DestroyVector2D(weight_update_container[thread][i]);
       DestroyVector2D(bias_update_container[thread][i]);
    }




    }
    train_end = clock();
    printf("\nIteration %d - Total Error : %f\n", iteration, total_error);
    printf("\nIteration %d - Error whole error calculation + training time : %f\n\n", iteration, (train_end - train_start) / (double)CLOCKS_PER_SEC);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    printf ("Total time = %f seconds\n",
            (end.tv_nsec - begin.tv_nsec) / 1000000000.0 +
            (end.tv_sec  - begin.tv_sec));

    if(min_error > total_error)
    {
        min_error = total_error;

    for(int weight_index=0; weight_index < mlp_layer_count - 1;weight_index++)
    {
       // DestroyVector2D(best_weight_array[weight_index]);
       // DestroyVector2D(best_bias_array[weight_index]);

        best_weight_array[weight_index] = CopyVector(weight_array[weight_index]);
        best_bias_array[weight_index] = CopyVector(bias_array[weight_index]);
    }

    }



    }
    // We are copying best weight and biases to MLP...
    printf("\n\nMLP will be configured to best value with error of %f\n\n", min_error);
     for(int weight_index=0; weight_index < mlp_layer_count - 1;weight_index++)
    {

        weight_array[weight_index] = CopyVector(best_weight_array[weight_index]);
        bias_array[weight_index] = CopyVector(best_bias_array[weight_index]);
    }



    // When training finishes... We are cheching test error...

    Vector2D * final_output = FeedForward(test_data, 0);
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


// Finally we are measure how much time has elapsed during the program execution...
clock_t end_time = clock();
double time_spent = (double)(end_time - begin_time) / CLOCKS_PER_SEC;
printf("\n\nElapsed exact time %f", time_spent);

    return 0;
}
