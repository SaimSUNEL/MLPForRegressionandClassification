#ifndef MLPCLASSIFIER_H_INCLUDED
#define MLPCLASSIFIER_H_INCLUDED
/*
    This file contains the required functions related a Multilayer Perceptron for Classification tasks...

    Implemented Functions:
        float generate_uniform(float a, float b)
        void CreateMLP(int layer_count, ...)
        Vector2D * CreateWeightMatrix(int input_count, int output_count)
        Vector2D * FeedForward(Vector2D * input)
        Vector2D * sigmoid(Vector2D * vector)
        Vector2D * CreateOneHot(Vector2D * indexes, int vector_length)
        Vector2D * softmax(Vector2D * vector)
        void BackPropagate(Vector2D * input, Vector2D * labels)
        double MLPClassifierError(Vector2D * data_set, Vector2D * one_hot_labels)


*/









Vector2D * softmax(Vector2D * vector);

Vector2D * CreateWeightMatrix(int input_count, int output_count);
Vector2D * sigmoid(Vector2D * vector);

/*
    This function produces samples with uniform distribution

    Function is taken from https://bytes.com/topic/c/answers/576389-generating-random-numbers-uniform-b
*/
float generate_uniform(float a, float b)
{
return rand() / (RAND_MAX + 1.0) * (b - a) + a;
}


Vector2D ** best_weight_array;
Vector2D ** best_bias_array;


// This variable points the array of the MLP weight vectors....
Vector2D ** weight_array;
// This variable keeps the array of the bias weights of the layers of MLP
Vector2D ** bias_array;

// This pointer holds an array which keeps the structure of the MLP
int * layer_structure = NULL;
// Total layer count of the MLP
int mlp_layer_count = 0;

// These three pointers are used in Backpropagation...
// Each thread needs to hold its feedforward result so we are creating and array of array of layer results...
Vector2D *** layer_results; // Keeps the output of each layer for a given batch in forward pass...


Vector2D *** weight_update_container;
Vector2D *** bias_update_container;


// Learning rate of the Gradient Descent...
float learning_rate = 0.0001;

// This functions creates a MLP depending on the parameters...
// The MLP structure is not fixed, user can define variable MLP structures...
// according the structure specified by the parameter, this function allocates weight matrices that will be used...
// First parameter is the total layer number of the MLP, subsequent values specify the layer's node number
/*
    CreateMLP(3, 5, 10 , 5) -> specifies a three layered MLP, 5 nodes in input layer, 10 nodes in hidden layer, 5 nodes in output layer
    CreateMLP(5, 4, 20, 10, 3) -> specifies a five layer MLP, 4 nodes input layer, 20 nodes in first hidden layer, 10 in second hidden layer, 3 nodes in output layer...
*/
// total thread numbers stores how many threads will work on the dataset...
void CreateMLP(float learning_rate_, int total_thread_number,int layer_count, ...)
{
    mlp_layer_count = layer_count;
    learning_rate = learning_rate_;

   // While training the error may fluctuate so we will keep the least errorous weights and biases...
   // When training finishes this best values will be copied to MLP...
    best_weight_array = malloc((layer_count-1)*sizeof(Vector2D *));
    best_bias_array = malloc((layer_count - 1 )*sizeof(Vector2D *));


    //These variables point an array of pointers of the weights...
    weight_array = malloc((layer_count-1)*sizeof(Vector2D *));
    bias_array = malloc((layer_count-1)*sizeof(Vector2D *));
     layer_results = malloc(total_thread_number*sizeof(Vector2D**));

     for(int a = 0 ; a < layer_count-1; a++)
     {
         best_bias_array[a] = malloc(sizeof(Vector2D*));
         best_weight_array[a] = malloc(sizeof(Vector2D*));

     }

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

    //This will hold the layer values after forward pass to be used in backpropagation...

    // These variables hold the changes that will be performed in backpropagation...



    // We are storing the MLP strucuture in a int array, MLP layer structure is used while creating the weight and bias matrices...
    layer_structure = malloc(layer_count*sizeof(int));

    // To process variable number parameters in C we have perform these operations...
    va_list ap;
    va_start(ap, layer_count);
    for(int a=0; a<layer_count;a++)
    {
        layer_structure[a] = va_arg(ap, int);
    }
    va_end(ap);

    // Finally creating the weight matrices and bias matrices of the specified MLP
    for(int i=0; i<layer_count-1;i++)
    {
        weight_array[i] = CreateWeightMatrix(layer_structure[i], layer_structure[i+1]);
        bias_array[i] = CreateWeightMatrix(1, layer_structure[i+1]);
    }
}

/*
    To eliminate diminishing or exploding gradient problem in training, Xavier initialization is be used...


                                                             o         o
                                                 o           o         o
    Suppose we have CreateMLP(4, 4, 6 , 5, 2) -> o  a(0)i,j  o a(1)j,k o a(2)k,m o
                                                 o           o         o         o
                                                 o           o         o
                                                             o
    a(l)i,j represents the weight on layer l to layer l+1
    For each layer node number starts from 1.
    i represents layer l node, j represents layer l+1 node
    Input layer is layer 0, first hidden layer is layer 1, second hidden layer is layer 2, output is layer 3

    a(0) = [a(0)1,1 a(0)1,2 a(0)1,3 a(0)1,4 a(0)1,5 a(0)1,6]
           [a(0)2,1 a(0)2,2 a(0)2,3 a(0)2,4 a(0)2,5 a(0)2,6]
           [a(0)3,1 a(0)3,2 a(0)3,3 a(0)3,4 a(0)3,5 a(0)3,6]
           [a(0)4,1 a(0)4,2 a(0)4,3 a(0)4,4 a(0)4,5 a(0)4,6]

    a(1) = [a(1)1,1 a(1)1,2 a(1)1,3 a(1)1,4 a(1)1,5]
           [a(1)2,1 a(1)2,2 a(1)2,3 a(1)2,4 a(1)2,5]
           [a(1)3,1 a(1)3,2 a(1)3,3 a(1)3,4 a(1)3,5]
           [a(1)4,1 a(1)4,2 a(1)4,3 a(1)4,4 a(1)4,5]
           [a(1)5,1 a(1)5,2 a(1)5,3 a(1)5,4 a(1)5,5]
           [a(1)6,1 a(1)6,2 a(1)6,3 a(1)6,4 a(1)6,5]

    a(2) = [a(2)1,1 a(2)1,2]
           [a(2)2,1 a(2)2,2]
           [a(2)3,1 a(2)3,2]
           [a(2)4,1 a(2)4,2]
           [a(2)5,1 a(2)5,2]
           [a(2)6,1 a(2)6,2]

    Besides those weights there are also bias weights added in the computation...
    b(0) = [b(0)1 b(0)2 b(0)3 b(0)4 b(0)5 b(0)6]
    b(1) = [b(1)1 b(1)2 b(1)3 b(1)4 b(1)5]
    b(2) = [b(2)1 b(2)2]
    Those weights are created for layer of the MLP...


*/
Vector2D * CreateWeightMatrix(int input_count, int output_count)
{
    // Every weights between layers are represented as matrices...
    // We are creating weight matrix....
    float init_range = 0;
    Vector2D * temp = malloc(sizeof(Vector2D));
    temp->height = input_count; //For bias...
    temp->width = output_count;
    temp->data = malloc(sizeof(float)*(input_count+1)*output_count);


    // This initialization assumes number of inputs = number of outputs, for simplicity it is assumed so..
    init_range = sqrt(3.0 / input_count);

    // We filling our weight matrix will uniformly distributed samples....
    for(int a=0; a<input_count*output_count; a++)
    {
        temp->data[a] = generate_uniform(-init_range, init_range);
    }
    return temp;
}


// Performs forward propagation for the MLP
// Input can be whole data or batch data... This functions calculates outputs of each layer and stores the results in a global array...
// Output result is returned
// As activation logistic function(sigmoid) is used...
// Immediate layer's results will be used in backpropagation, therefore it is stored...
/*
    matmul = matrix multiplication...

    output_layer(0) = sigmoid(matmul(input, a(0))+b(0))
    output_layer(1) = sigmoid(matmul(output_layer(0), a(1))+b(1))

    output_layer(2) is the output, it just performs summation operation of the previous layer's layer..
    and result is passed to softmax operation to calculate the probabilities...

    output_layer(2) = softmax(matmul(output_layer(1), a(2))+b(2))



*/

Vector2D * FeedForward(Vector2D * input, int thread_number)
{   Vector2D * temp ;


    // If there are more than sample in the input, the output of every layer is a matrix
    // But the bias vector is just one vector... It should be added to every row of the output of layer
    // To be able perform this operation, every bias vector is multiplied with a vector consisting of ones...
    // By douing so a matrix form is obtained and this matrix is added to result

    /*
        Suppose a(0) = [1 2 3]  and input = [1, 5]  b(0)= [2 3]   matmul(input, a(0)) = [23 35]
                       [4 5 6]              [5, 6]                                      [73 86]
                                            [4, 6]
        So for each sample the b(0) should be added to every row of matmul result
        So we can do it by matrix multiplied by [1] column vector matmul([1], b(0)) = [2 3] by so it can be added..
                                                [1]                      [1]          [2 3]

        1 is used because every layer 0th node is assumed to be 1 while adding bias...

    */
    float * ones_ = malloc(sizeof(float)*input->height);
    for(int i=0;i<input->height;i++)ones_[i] = 1.0;
    Vector2D * ones = CreateVector2D(ones_, input->height, 1);
    Vector2D * bias;



    // By staring from first layer we are  performing forward pass iteratively...
    for(int current_layer=1; current_layer < mlp_layer_count; current_layer++)
    {
        {
             // weight matrix of the layer and the previous input is multiplied
             temp = MatrixProduct(input, weight_array[current_layer-1]);
             // Bias matrix is obtained...
             bias = MatrixProduct(ones, bias_array[current_layer-1]);
             // The bias is added to matmul operation...
             temp = MatrixAdd(temp, bias);
             DestroyVector2D(bias);

             // If we are at output layer the hidden layer will be passed through the sigmoid function...
            if(current_layer < mlp_layer_count -1)
                input = sigmoid(temp);
            // If at output we are softmaxing the last hidden layer output
            else
                input = softmax(temp);
       }

       // Every layer output is kept....
       layer_results[thread_number][current_layer-1] = input;
    }

    // The output layer's result is returned...
    return layer_results[thread_number][mlp_layer_count-2];
}

// This function performs sigmoid function on each element of matrix/vector and overrides the original vector/matrix
/*
    ^ operator is used for power operation

    sigmoid(x) = 1/(1 + e^(-x))
*/

Vector2D * sigmoid(Vector2D * vector)
{
    for(int i=0; i < vector->height*vector->width; i++)
    {

        vector->data[i] = (1.0/(1.0+exp(-vector->data[i])));
    }
    return vector;

}


/*
    While performing the classification operation, our networks output produces the output probability of each class of our problem..
    While training MLP for classificiation, ground truth is represent as one hot vector representation...
    Only one element is 1 and the others are zero...

    Suppose we have 3 classes to identify with MLP. A, B, C.
    A is represented via [1 0 0] vector, B [0 1 0], [0 0 1]
    Our MLP output consists of 3 nodes...
    Because we are using softmax, the each output node value indicates the probability of the sample belonging to that class...
    First node outputs the probability of the given sample belonging to A class
    Second node for B class, Third node for C class. And while determining a sample's estimated class, we are choosing the one whose probability values is the highest...

    This function takes index values(those which will be 1 in one hot vector representation) and one hot vector's length
    indexes = [1]    CreateOneHot(indexes, 4) -> [0 1 0 0]
              [2]                                [0 0 1 0]
              [3]                                [0 0 0 1]
              [0]                                [1 0 0 0]
*/
// A new matrix is created to store one hot vector representation
Vector2D * CreateOneHot(Vector2D * indexes, int vector_length)
{
    //
    Vector2D * one_hot_vector = malloc(sizeof(Vector2D));
    one_hot_vector->height = indexes->height;
    one_hot_vector->width = vector_length;
    one_hot_vector->size = one_hot_vector->height;
    one_hot_vector->data = malloc(sizeof(float)*indexes->height*vector_length);

    // All values are zeroed....
    memset(one_hot_vector->data, 0, sizeof(float)*indexes->height*vector_length);
    // We are turning the indicated index to one in indexes variable...
    for(int i=0; i<one_hot_vector->height;i++)
    {
        one_hot_vector->data[i*vector_length+(int)indexes->data[i*indexes->width]] = 1.0;
    }
    return one_hot_vector;
}

/*
    This functions performs softmax function for each row of the given matrix/vector...
    The original is overrided with the new result....

    softmax(xi) = e^(xi)/sum(for x)
*/
Vector2D * softmax(Vector2D * vector)
{
    float sum = 0;
    // Exponent of each element in a row is stored in this array, because every element should be visited in row vector...
    float * temp = malloc(sizeof(float)*vector->width);

    for(int h=0; h<vector->height;h++)
    {

        sum = 0;
        for(int row=0; row < vector->width; row++)
        {   // We are storing each element of the row vector and calculating the summation of them....
            temp[row] = exp(vector->data[h*vector->width+row]);
            sum += temp[row];
        }

        // Later on each row element is overrided with its exponent divided by the all exponents' summation...
        for(int row=0; row < vector->width; row++)
        {
            vector->data[h*vector->width+row] = temp[row]/sum;
        }

    }
    free(temp);
    return vector;
}

/*
    This function performs backpropagation algoritm and updates the weight matrices of the MLP
    It first applies forward propagation to obtain the intermediate layer outputs....
    Error signal is propagated towards back, and each weight adjusts itself depending on contribution of it to error..
    Error signal starts at output and passes to last hidden later up to inputs propagates...

    Obtained rules are:

    Δa(2) = η∑matmul(Transpose(X(2)).Err(output))
    Δa(1) = η∑ matmul(Transpose(X1), Pairwise(matmul(Err(output), Transpose(a(2))), Pairwise(X(2), 1-X(2))))
    Δa(0) = η∑ matmul(Transpose(intput), Paiwise( matmul(Pairwise(matmul(Err(output), Transpose(a(2))), Pairwise(X(2), 1-X(2))), Transpose(a(1))), Pairwise((X1), 1-X(1))))

    In general form:
    Δa(l) = η∑ matmul(Transpose(X(l)), Pairwise(matmul(Err(l+1), Transpose(a(l+1))), Pairwise(X(l+1), 1-X(l+1))))

*/



void BackPropagate(Vector2D * input, Vector2D * labels, int thread_number)
{
    // Firstly we are getting the outputs of each layer...
    Vector2D * output = FeedForward(input, thread_number);


    Vector2D * labels_copy = CopyVector(labels);


    // We are calculating the error at outout....
    Vector2D * err = MatrixSubtract(labels_copy, output);

    // While updating biases, a row column of ones is required....
    float * ones_ = malloc(sizeof(float)*input->height);
    for(int i=0;i<input->height;i++)ones_[i] = 1.0;
    Vector2D * ones = CreateVector2D(ones_, 1, input->height);


    // Lots of pointers are used because immediately created vector/matrix should be cleared from the memory...
    // To reduce the matrix amount used in program... Otherwise allocated memory will increase by the time passes...

    Vector2D * scalar_minus;

    DestroyVector2D(output);
    Vector2D * temp, * temp1, * temp2, *slice, * transpose;

    // We are starting from output layer...
    for( int current_layer = mlp_layer_count-2; current_layer>0;current_layer--)
    {
        // if we are output layer its weight should be adjust by simply
        // performing matrix multiplication with the previous layer output and output errpr...
        if(current_layer == mlp_layer_count-2)
        {
            // The previous layer's output is transposed...
            temp = TransposeVector2D(layer_results[thread_number][current_layer-1]);

            // The bias also must be updated similar to normal weights
            /*
                We could have inserted the bias weights to normal layer weights matrix but in intermediate
                layer while performing calculation column of 1 should be added and this brings some more extra work in memory..
                If it weren't seperated the output of the previous layer output would be X(2) = [1 2 3]
                                                                                                [1 4 5]
                                                                                                [1 6 7]
                while updating the weight we are taking transpose of the previous layer so that is why for seperate
                bias updates we are performing matrix multiplication with bias. If we transpose the output above
                we would have obtained [1 1 1] this one column vector comes from this fact...
                                       [2 4 6]
                                       [1 6 7]
            */
            // We are storing the updates in an array of pointer later we will update our actual weights...
            weight_update_container[thread_number][current_layer] = MatrixProduct(temp, err);
            bias_update_container[thread_number][current_layer] = MatrixProduct(ones, err);



            DestroyVector2D(temp);
           // printf("\nLast layer weights has been updated...\n");
            continue;
        }

        // If we are in hidden layer,

        temp1 = err ;
        transpose = TransposeVector2D(weight_array[current_layer+1]);

        // Error propagated to current layer is obtained by multiplying next layer's error with transpose of next layer's weight matrix
        // later pairwisely multiplying the output of next layer and with 1 - next layer's output....
       // error is propagated to next layer
       err = MatrixProduct(temp1, transpose);
       DestroyVector2D(transpose);
       DestroyVector2D(temp1);

        scalar_minus = ScalarMinusVector2D(1, layer_results[thread_number][current_layer]);
        // When we multiply prev error with the next layer's output
        /*
            next layer output was obtained via sigmoid, derivative of the sigmoid is (sigmoid*(1-sigmoid))
            While applying chaing rule this term takes places in the backprogation...
        */
        // So multiplying it we are exactly calculating the next layer's error...
        err = MatrixPairwiseProduct(MatrixPairwiseProduct(err, layer_results[thread_number][current_layer]), scalar_minus);
       // printf("\n\nSUNEL\n\n");
       DestroyVector2D(scalar_minus);

        // We are transposing the prev layer output to calculate current layer's weight change...
        transpose = TransposeVector2D(layer_results[thread_number][current_layer - 1 ]);

        weight_update_container[thread_number][current_layer] = MatrixProduct( transpose, err);
        // Bias is also updated according to explanation above....
        bias_update_container[thread_number][current_layer] = MatrixProduct(ones, err);
        DestroyVector2D(transpose);

    }

    // Finally we are in the input layer, nothing changes just index of the variables are specified explicitly....
    temp1 = err;


     // First layer error is calculated...
     transpose = TransposeVector2D(weight_array[1]);

     err = MatrixProduct(temp1, transpose);
     DestroyVector2D(transpose);
    DestroyVector2D(temp1);
    scalar_minus = ScalarMinusVector2D(1, layer_results[thread_number][0]);
     err = MatrixPairwiseProduct(MatrixPairwiseProduct(err, layer_results[thread_number][0]), scalar_minus);//VectorSlice2D(layer_results[0], 0, layer_results[0]->height, 1, layer_results[0]->width));
     DestroyVector2D(scalar_minus);

     // Because there is no previous layer, the input is transposed...
     transpose = TransposeVector2D(input);

     // weight updates are obtained...
     weight_update_container[thread_number][0] = MatrixProduct(transpose, err);
     bias_update_container[thread_number][0] = MatrixProduct(ones, err);
     DestroyVector2D(transpose);
     DestroyVector2D(err);



    //Wegiht update is performed after every thread finishes it job

}





/*
    This functions calculates cross entropy loss function with a given samples and its corresponding labels...
    Cross_Entropy(x_set) = −∑ lable(i)*log(sample(i))
    labels are one hot vectors.
    this operation multiplies two vector pairwisely and the result vector's elements are summed...

    log(output) = [0.2 0.3 0.5] one_hot_labels = [0 1 0]  output*one_hot_labels =[0.2*0 0.3*1 0.5*0]  ->[0 0.3 0]
                  [0.4 0.1 0.5]                  [1 0 0]                        [0.4*1 0.1*0 0.5*0]    [0.4 0 0]
                  [0.8 0.1 0.1]                  [0 0 1]                        [0.8*0 0.1*0 0.1*1]    [0 0 0.1]

    And this result is summed... 0.3 + 0.4 + 0.1=0.8

*/

double MLPClassifierError(Vector2D * data_set, Vector2D * one_hot_labels, int thread_id)
{
    Vector2D * output, * log_output, * pairwise;
    double result = 0;
    // we are obtaining the output of MLP for given data set...
    output = FeedForward(data_set, thread_id);

    // We are taking logarithm of output...
    log_output = Log2D(output);
    Vector2D * one_hot_labels_copy = CopyVector(one_hot_labels);
    // We are multiplying labels with log of outputs...
    pairwise = MatrixPairwiseProduct(one_hot_labels_copy, log_output);
    DestroyVector2D(log_output);
    // And we are summing over whole error matrix to get a scalar value...
    result = -Sum2D(pairwise);
    DestroyVector2D(pairwise);
    return result;
}


#endif // MLPCLASSIFIER_H_INCLUDED
