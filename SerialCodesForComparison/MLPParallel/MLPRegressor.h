#ifndef MLPREGRESSOR_H_INCLUDED
#define MLPREGRESSOR_H_INCLUDED
Vector2D * softmax(Vector2D * vector);

Vector2D * CreateWeightMatrix(int input_count, int output_count);
Vector2D * sigmoid(Vector2D * vector);


float generate_uniform(float a, float b)
{
return rand() / (RAND_MAX + 1.0) * (b - a) + a;
}


Vector2D ** best_weight_array;
Vector2D ** best_bias_array;


Vector2D ** weight_array;
Vector2D ** bias_array;
int * layer_structure = NULL;
int mlp_layer_count = 0;

Vector2D ** layer_results;
Vector2D ** layer_updates;
Vector2D ** bias_updates;


Vector2D *** layer_results; // Keeps the output of each layer for a given batch in forward pass...


Vector2D *** weight_update_container;
Vector2D *** bias_update_container;



float learning_rate = 0.00001;

void CreateMLP(float learning_rate_, int total_thread_number, int layer_count, ...)
{
    mlp_layer_count = layer_count;
    learning_rate = learning_rate_;

    best_weight_array = malloc((layer_count-1)*sizeof(Vector2D *));
    best_bias_array = malloc((layer_count - 1 )*sizeof(Vector2D *));

    weight_array = malloc((layer_count-1)*sizeof(Vector2D *));
    bias_array = malloc((layer_count-1)*sizeof(Vector2D *));

    //This will hold the layer values afte forward pass to be used in backpropagation...

      for(int a = 0 ; a < layer_count-1; a++)
     {
         best_bias_array[a] = malloc(sizeof(Vector2D*));
         best_weight_array[a] = malloc(sizeof(Vector2D*));

     }

    layer_results = malloc(total_thread_number*sizeof(Vector2D**));

    /*
        Each thread will run forward and backpropagation, therefore the results that will be used in backpropagation
        should be stored for each thread... So we are to update weights and bias weights...
        Each thread will calculate its bias and weight update and put these into a global array...
        After each thread finishes its job, the stored update values for each thread will be used to update
        weights and biases.
    */
    weight_update_container = malloc(sizeof(Vector2D **)*total_thread_number);
    bias_update_container = malloc(sizeof(Vector2D **)*total_thread_number);
     for(int a = 0; a < total_thread_number; a++)
     {
         weight_update_container[a] = malloc(sizeof(Vector2D*)*(layer_count-1));
         layer_results[a] = malloc(sizeof(Vector2D*)*(layer_count-1));
         bias_update_container[a] = malloc(sizeof(Vector2D*)*(layer_count-1));



     }







    layer_structure = malloc(layer_count*sizeof(int));
    va_list ap;
    va_start(ap, layer_count);
    for(int a=0; a<layer_count;a++)
    {
        layer_structure[a] = va_arg(ap, int);
    }
    va_end(ap);

    for(int i=0; i<layer_count-1;i++)
    {
        weight_array[i] = CreateWeightMatrix(layer_structure[i], layer_structure[i+1]);
        bias_array[i] = CreateWeightMatrix(1, layer_structure[i+1]);
    }
}

/*
Xavier He initialization will be used...
*/
Vector2D * CreateWeightMatrix(int input_count, int output_count)
{
    float init_range = 0;
    Vector2D * temp = malloc(sizeof(Vector2D));
    temp->height = input_count; //For bias...
    temp->width = output_count;
    temp->data = malloc(sizeof(float)*(input_count+1)*output_count);

    init_range = sqrt(2.0 / input_count);

    for(int a=0; a<input_count*output_count; a++)
    {
        temp->data[a] = generate_uniform(-init_range, init_range);
    }
    return temp;
}


//Performs forward propagation for the MLP
Vector2D * FeedForward(Vector2D * input, int thread_number)
{  Vector2D * temp ;
    float * ones_ = malloc(sizeof(float)*input->height);
    for(int i=0;i<input->height;i++)ones_[i] = 1.0;
    Vector2D * ones = CreateVector2D(ones_, input->height, 1);
    Vector2D * bias;
    //While counting we exclude input layer...
    for(int current_layer=1; current_layer < mlp_layer_count; current_layer++)
    {
        //If we are not at the last layer, we are calculation the hidden layer's output by
        //multiplying the previous layer output with weight matrix before current layer
        // Then we are applying sigmoid and adding +1 to result...
        //printf("Current layer %d\n", current_layer-1);
        if(current_layer < mlp_layer_count -1)
        {
             temp = MatrixProduct(input, weight_array[current_layer-1]);
             bias = MatrixProduct(ones, bias_array[current_layer-1]);
             temp = MatrixAdd(temp, bias);
             DestroyVector2D(bias);
            // if(current_layer != 1) DestroyVector2D(input);
            // printf("\nMatrixProductRow add : \n");
            // DisplayVector2D(input);
            // printf("\nSigmoid applied : \n");
             input = sigmoid(temp);
            // DisplayVector2D(input);
        }

       else //In last layer, we are just accumulation previous layer outputs and applying softmax....
       { //printf("\nLast layer : \n");
           temp = MatrixProduct(input, weight_array[current_layer-1]);
            bias = MatrixProduct(ones, bias_array[current_layer-1]);
             temp = MatrixAdd(temp, bias);
             DestroyVector2D(bias);
          // DestroyVector2D(input);
           //printf("\nMatrix product : \n");
           //DisplayVector2D(input);
           input = temp;
           //printf("\nSoftmax applied : \n");
          //DisplayVector2D(input);
       }


       layer_results[thread_number][current_layer-1] = input;
    }


    return layer_results[thread_number][mlp_layer_count-2];
}

Vector2D * sigmoid(Vector2D * vector)
{
    for(int i=0; i < vector->height*vector->width; i++)
    {

        if((i % vector->width) != 0)
        vector->data[i] = (1.0/(1.0+exp(-vector->data[i])));
    }
    return vector;

}



void BackPropagate(Vector2D * input, Vector2D * labels, int thread_number)
{

    Vector2D * output = FeedForward(input);
    Vector2D * labels_copy = CopyVector(labels);
    Vector2D * err = MatrixSubtract(labels_copy, output);

    float * ones_ = malloc(sizeof(float)*input->height);
    for(int i=0;i<input->height;i++)ones_[i] = 1.0;
    Vector2D * ones = CreateVector2D(ones_, 1, input->height);


    Vector2D * scalar_minus;

    DestroyVector2D(output);
    Vector2D * temp, * temp1, * temp2, *slice, * transpose;

    //printf("Error is okay...");
   // printf("\nmlp_layer_count : %d\n", mlp_layer_count);
    for( int current_layer = mlp_layer_count-2; current_layer>0;current_layer--)
    {
       // printf("\nCurrent layer = %d\n", current_layer);
        if(current_layer == mlp_layer_count-2)
        {   //printf("\nLast layer weights will be updated...\n");
            temp = TransposeVector2D(layer_results[thread_number][current_layer-1]);

            weight_update_container[thread_number][current_layer] = MatrixProduct(temp, err);
            bias_update_container[thread_number][current_layer] = MatrixProduct(ones, err);



            DestroyVector2D(temp);
           // printf("\nLast layer weights has been updated...\n");
            continue;
        }
        temp1 = err ;
        transpose = TransposeVector2D(weight_array[current_layer+1]);


        err = MatrixProduct(temp1, transpose);
       DestroyVector2D(transpose);
       DestroyVector2D(temp1);


        scalar_minus = ScalarMinusVector2D(1, layer_results[thread_number][current_layer]);

        err = MatrixPairwiseProduct(MatrixPairwiseProduct(err, layer_results[thread_number][current_layer]), scalar_minus);
       // printf("\n\nSUNEL\n\n");
       DestroyVector2D(scalar_minus);


        transpose = TransposeVector2D(layer_results[thread_number][current_layer - 1 ]);

        weight_update_container[thread_number][current_layer] = MatrixProduct( transpose, err);
        bias_update_container[thread_number][current_layer] = MatrixProduct(ones, err);
        DestroyVector2D(transpose);

    }

    temp1 = err;

    transpose = TransposeVector2D(weight_array[1]);
     err = MatrixProduct(temp1, transpose);
     DestroyVector2D(transpose);
     DestroyVector2D(temp1);


    scalar_minus = ScalarMinusVector2D(1, layer_results[thread_number][0]);

     err = MatrixPairwiseProduct(MatrixPairwiseProduct(err, layer_results[thread_number][0]), scalar_minus);//VectorSlice2D(layer_results[0], 0, layer_results[0]->height, 1, layer_results[0]->width));
     DestroyVector2D(scalar_minus);


     transpose = TransposeVector2D(input);

     weight_update_container[thread_number][0] = MatrixProduct(transpose, err);
     bias_update_container[thread_number][0] = MatrixProduct(ones, err);
     DestroyVector2D(transpose);
     DestroyVector2D(err);

}


// this function calculates squared error between output and the target...

double MLPRegressorError(Vector2D * data_set, Vector2D * one_hot_labels)
{
    Vector2D * output, * error, * pairwise;
    double result = 0;
    output = FeedForward(data_set);
    Vector2D * one_hot_labels_copy = CopyVector(one_hot_labels);
    error = MatrixSubtract(one_hot_labels_copy, output);
    error = MatrixPairwiseProduct(error, error);
    result = 0.5*Sum2D(error);
    DestroyVector2D(error);
    DestroyVector2D(output);
    return result;
}


#endif // MLPREGRESSOR_H_INCLUDED
