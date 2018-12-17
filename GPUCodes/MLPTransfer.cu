#include<stdio.h>
#include<string.h>
#include <stdlib.h>
#include <stdarg.h>
#include<time.h>
#include <math.h>
#define CHECK(call) \
{ \
 const cudaError_t error = call; \
 if (error != cudaSuccess) \
 { \
 printf("Error: %s:%d, ", __FILE__, __LINE__); \
 printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
 exit(1); \
 } \
}


// C language does not contain a boolean type, we are defining our type...
#define FALSE 0
#define TRUE 1


// All vectors/matrices are stored as this structure in the memory...
struct Vector2D
{
    // Whole vector/matrix data is stored in one dimensional array...
    // All numbers are floating point numbers....

    //This pointer points where the vector/matrix data lyies....
    float * data;
    // Row number of the vector/matrix...
    int height;
    // Column number of the vector/matrix...
    int width;

    int size;

};

// We are defining a type from this structure definition...
typedef struct Vector2D Vector2D;



__device__ Vector2D array[5];

float * device_matrix_location;

Vector2D * CreateVector2D(float * data, int height, int width, bool fill = true, bool store = false)
{
    // A new structure is allocated in GPU memory for matrix/vector...
    Vector2D * temp ;
    
    CHECK(cudaMalloc(&temp, sizeof(Vector2D)));
    float * temp2;
    CHECK(cudaMalloc(&temp2, sizeof(float)*height*width));
    if(fill == true)
    CHECK(cudaMemcpy(temp2, data, sizeof(float)*height*width, cudaMemcpyHostToDevice));
    
    CHECK(cudaMemcpy(&temp->data, &temp2, sizeof(float *), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(&temp->height, (void *)(&height), sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(&temp->width, (void *)(&width), sizeof(int), cudaMemcpyHostToDevice));
    //temp->height = height;
    //temp->width = width;
    if(store == true)
    device_matrix_location = temp2;
    cudaDeviceSynchronize();
    return temp;
}



__global__ void MatrixAdd(Vector2D * result, Vector2D * vec1, Vector2D * vec2)
{
	if((vec1->width != vec2->width) ||   (vec1->height != vec2->height))
	{
		printf("\n\n**********Matrix add diff dimension....");
		return;
	}
	
	
	int tx = blockIdx.x*blockDim.x+ threadIdx.x;
	int ty = blockIdx.y*blockDim.y + threadIdx.y;
	int tid = ty*vec1->width+tx;
	if(tid ==0)
	{//printf("\nMatrixAddvec->width : %d vec->height : %d - x dim %d y dim %d\n", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	
	if((blockDim.y*gridDim.y < vec1->height) || (blockDim.x*gridDim.x<vec1->width))
	{
		printf("\nMatrixAdd\n");
		printf("vec->width : %d vec->height : %d - x dim %d y dim %d", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	} 
}
	
	if(tid < vec1->width*vec1->height)
	{
		result->data[tid] = vec1->data[tid] + vec2->data[tid];
	}
	
	
}

__global__ void MatrixSubtract(Vector2D * result, Vector2D * vec1, Vector2D * vec2)
{
	
		if((vec1->width != vec2->width) ||   (vec1->height != vec2->height))
	{
		printf("\n\n**********Matrix Subtract diff dimension....");
		return;
	}
	
	
	int tx = blockIdx.x*blockDim.x+ threadIdx.x;
	int ty = blockIdx.y*blockDim.y + threadIdx.y;
	int tid = ty*vec1->width+tx;
	
	if(tid ==0)
	{
		//printf("\nMatrixSubtractvec->width : %d vec->height : %d - x dim %d y dim %d\n", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	
	if((blockDim.y*gridDim.y < vec1->height) || (blockDim.x*gridDim.x<vec1->width))
	{
		printf("\nMatrixSubtract\n");
		printf("vec->width : %d vec->height : %d - x dim %d y dim %d", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	} 
}
	
	if(tid < vec1->width*vec1->height)
	{
		result->data[tid] = vec1->data[tid] - vec2->data[tid];
	}
	
	
}



__global__ void TransposeVector2D(Vector2D * res, Vector2D * m1)
{
	
		if((res->width != m1->height) ||   (res->height != m1->width))
	{
		printf("\n\n**********Matrix Transpose diff dimensionç....");
		printf("\nres->width : %d res->heihgt : %d - m1->width : %d m1->height %d\n", res->width, res->height, m1->width, m1->height);
		return;
	}
	
	
	int thx = blockIdx.x*blockDim.x+ threadIdx.x;
	int thy = blockIdx.y*blockDim.y+threadIdx.y;
	int tid = thx + thy*m1->width;
	
	if(tid ==0)
	{
		//printf("\nTransposeVector2Dvec->width : %d vec->height : %d - x dim %d y dim %d\n", m1->width, m1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	
	if((blockDim.y*gridDim.y < m1->height) || (blockDim.x*gridDim.x<m1->width))
	{
		printf("\nTransposeVector2D\n");
		printf("vec->width : %d vec->height : %d - x dim %d y dim %d", m1->width, m1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	} 
}
	
	
	
	if(tid < m1->width*m1->height)
	{
	
	res->data[thy+thx*m1->height] = m1->data[tid] ;
	
	//printf("idy : %d - idx : %d - blockdim x : %d - blockDim y : %d - gridDim.x - %d - gridDim.y : %d\n", thy, thx, blockDim.x, blockDim.y, gridDim.x, gridDim.y);
	}
	
}

__global__ void DisplayVector2D(Vector2D * vector)
{
    printf("[");
    for(int h = 0; h < vector->height; h++)
    {
        printf("[");
        for( int w = 0; w < vector->width-1; w++)
        {
            printf("%f, ", vector->data[h*vector->width+w]);
        }
        printf("%f], \n", vector->data[h*vector->width+vector->width-1]);
    }
    printf("]\n");
    
    printf("Row : %d - Width : %d \n\n", vector->height, vector->width);
    
}



__global__ void MatrixProduct(Vector2D * result, Vector2D * m1, Vector2D * m2)
{
	
	
	
	int thx = blockIdx.x*blockDim.x+ threadIdx.x;
	int thy = blockIdx.y*blockDim.y+threadIdx.y;
	
	if(thx == 0 && thy ==0){
		if((m1->width != m2->height))
	{
		printf("\n\n**********Matrix Product error dimensionç....");
		printf("\nm1->width %d m1->height %d - m2->width %d m2->height %d\n", m1->width, m1->height, m2->width, m2->height);
		return;
	}
	//printf("\nMatrixProductvec->width : %d vec->height : %d - x dim %d y dim %d\n", result->width, result->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	
	if(thx ==0 && thy == 0)
	if((blockDim.y*gridDim.y < result->height) || (blockDim.x*gridDim.x<result->width))
	{
		printf("\nMatrixProduct\n");
		printf("vec->width : %d vec->height : %d - x dim %d y dim %d", result->width, result->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	} 
	

}
	
	
	
	if(thx < result->width && thy < result->height)
	{
		float toplam = 0;
		for(int h = 0; h < m1->width; h++)
		{
			toplam += m1->data[thy*m1->width+h] * m2->data[h*m2->width+thx];
		}
	result->data[thy*result->width + thx] = toplam;
	//printf("idy : %d - idx : %d - blockdim x : %d - blockDim y : %d - gridDim.x - %d - gridDim.y : %d\n", thy, thx, blockDim.x, blockDim.y, gridDim.x, gridDim.y);
	}
	
}


__global__ void ScalarMinusVector2D(Vector2D * result, float value, Vector2D * vec1)
{
		if((result->width != vec1->width) ||   (result->height != vec1->height))
	{
		printf("\n\n**********Scaar Minus vectrordiff dimensionç....");
		return;
	}
	int tx = blockIdx.x*blockDim.x+ threadIdx.x;
	int ty = blockIdx.y*blockDim.y + threadIdx.y;
	int tid = ty*vec1->width+tx;
	
	if(tid ==0)
	{
		//printf("\nScalarMinusVector2Dvec->width : %d vec->height : %d - x dim %d y dim %d\n", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	
	if((blockDim.y*gridDim.y < vec1->height) || (blockDim.x*gridDim.x<vec1->width))
	{
		printf("\nScalarMinusVector2D\n");
		printf("vec->width : %d vec->height : %d - x dim %d y dim %d", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	} 
}
	
	if(tid < vec1->width*vec1->height)
	{
		result->data[tid] = value-vec1->data[tid];
	}
	
	
}

__global__ void ScalarMatrixProduct(Vector2D * result, float scalar, Vector2D * vec1)
{
	
		if((result->width != vec1->width) ||   (result->height != vec1->height))
	{
		printf("\n\n**********ScalarMatrixProduct dimensionç....");
		return;
	}
	int tx = blockIdx.x*blockDim.x+ threadIdx.x;
	int ty = blockIdx.y*blockDim.y + threadIdx.y;
	int tid = ty*vec1->width+tx;
	
	if(tid ==0)
	{
		//printf("\nScalarMatrixProductvec->width : %d vec->height : %d - x dim %d y dim %d\n", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	
	if((blockDim.y*gridDim.y < vec1->height) || (blockDim.x*gridDim.x<vec1->width))
	{
		printf("\nScalarMatrixProduct\n");
		printf("vec->width : %d vec->height : %d - x dim %d y dim %d", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	} 
}
	
	if(tid < vec1->width*vec1->height)
	{
		result->data[tid] = scalar*vec1->data[tid];
	}
}


__global__ void MatrixPairwiseProduct(Vector2D * result, Vector2D * vec1, Vector2D * vec2)
{
	
	
		if((vec1->width != vec2->width) ||   (vec1->height != vec2->height))
	{
		printf("\n\n**********MatrixPairwiseProduct dimension....");
		return;
	}
	
	
	
	int tx = blockIdx.x*blockDim.x+ threadIdx.x;
	int ty = blockIdx.y*blockDim.y + threadIdx.y;
	int tid = ty*vec1->width+tx;
	
	
	if(tid ==0)
	{
		//printf("\nMatrixPairwiseProductvec->width : %d vec->height : %d - x dim %d y dim %d\n", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	
	if((blockDim.y*gridDim.y < vec1->height) || (blockDim.x*gridDim.x<vec1->width))
	{
		printf("\nMatrixPairwiseProduct\n");
		printf("vec->width : %d vec->height : %d - x dim %d y dim %d", vec1->width, vec2->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	} 
}
	
	
	if(tid < vec1->width*vec1->height)
	{
		result->data[tid] = vec1->data[tid] * vec2->data[tid];
	}
	
	
}
__device__ float error_sum[32];
__global__ void Sum2D(Vector2D * vec)
{
	int tid = threadIdx.y;
	int val = 0;
	int width = vec->width;
	
	for(int a = 0; a < width; a++)
	{
		val += vec->data[a+tid*width];
	}	
	error_sum[tid] = val;
}
__global__ void ArgMax2D(Vector2D * vector)
{
	
	
}

__global__ void Log2D(Vector2D * result, Vector2D * vec1)
{
	int tx = blockIdx.x*blockDim.x+ threadIdx.x;
	int ty = blockIdx.y*blockDim.y + threadIdx.y;
	int tid = ty*vec1->width+tx;
	
	
	
	if(tid ==0)
	{
		//printf("\nLog2Dvec->width : %d vec->height : %d - x dim %d y dim %d\n", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	
	if((blockDim.y*gridDim.y < vec1->height) || (blockDim.x*gridDim.x<vec1->width))
	{
		printf("\nLog2D\n");
		printf("vec->width : %d vec->height : %d - x dim %d y dim %d", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	} 
}
	
	
	
	float val;
	if(tid < vec1->width*vec1->height)
	{
		
		
		val = log(vec1->data[tid]);
		
		result->data[tid] = val;
		
	}
	
	
}

__global__ void Sigmoid(Vector2D * result, Vector2D * vec1)
{
	int tx = blockIdx.x*blockDim.x+ threadIdx.x;
	int ty = blockIdx.y*blockDim.y + threadIdx.y;
	int tid = ty*vec1->width+tx;
	
	
	
	if(tid ==0)
	{
		//printf("\nSigmoidvec->width : %d vec->height : %d - x dim %d y dim %d\n", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	
	if((blockDim.y*gridDim.y < vec1->height) || (blockDim.x*gridDim.x<vec1->width))
	{
		printf("\nSigmoid\n");
		printf("vec->width : %d vec->height : %d - x dim %d y dim %d", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	} 
}
	
	if(tid < vec1->width*vec1->height)
	{
		result->data[tid] = 1.0/(1.0 + exp(-(vec1->data[tid])));
	}
	
}

__global__ void PointerSet(Vector2D * f1, Vector2D * f2, int shift, int batch_size)
{
	f1->width = f2->width;
	f1->height = batch_size;
	f1->data = f2->data + f2->width*shift;	
}


/*
 * 
 * MLP functions....
 * 
 * 
 * 
 * 
 * 
 */





float generate_uniform(float a, float b)
{
return rand() / (RAND_MAX + 1.0) * (b - a) + a;
}


Vector2D ** mlp_structure_information;
Vector2D ** mlp_layer_output_structure;
Vector2D ** mlp_layer_bias_structure;


Vector2D ** weight_array;
int * layer_structure = NULL;
int mlp_layer_count = 0;

Vector2D ** layer_results;

Vector2D ** layer_updates;
Vector2D ** bias_array;
Vector2D ** bias_results;

Vector2D ** device_weight_array;

Vector2D ** layer_error_array;
Vector2D ** scalar_minus_array;

Vector2D ** bias_updates;



float learning_rate = 0.000001;
Vector2D * CreateWeightMatrix(int input_count, int output_count);

void CreateMLP(int layer_count, ...)
{
    mlp_layer_count = layer_count;
    weight_array = (Vector2D **)malloc((layer_count-1)*sizeof(Vector2D *));
    bias_array = (Vector2D **)malloc((layer_count - 1 )*sizeof(Vector2D *));
    bias_results = (Vector2D **)malloc((layer_count - 1 )*sizeof(Vector2D *));
    
    bias_updates = (Vector2D **)malloc((layer_count-1)*sizeof(Vector2D));
    
    
    layer_error_array = (Vector2D **)malloc((layer_count-1)*sizeof(Vector2D));
    
    mlp_structure_information = (Vector2D **)malloc((layer_count-1)*sizeof(Vector2D));
    mlp_layer_output_structure = (Vector2D **)malloc((layer_count-1)*sizeof(Vector2D));
    
    
    //This will hold the layer values afte forward pass to be used in backpropagation...
    layer_results = (Vector2D **)malloc((layer_count-1)*sizeof(Vector2D*));
    layer_updates = (Vector2D **)malloc((layer_count-1)*sizeof(Vector2D*));


    layer_structure = (int *)malloc(layer_count*sizeof(int));
    va_list ap;
    va_start(ap, layer_count);
    for(int a=0; a<layer_count;a++)
    {
        layer_structure[a] = va_arg(ap, int);
    }
    va_end(ap);
    
    printf("\nMLP structure\n");
    for(int a=0; a<mlp_layer_count;a++)
		printf("%d ", layer_structure[a]);
		
	printf("\n\n");
    

    for(int i=0; i<layer_count-1;i++)
    {
		
		//printf("\n\nLayer %d\n", i);
		
        weight_array[i] = CreateWeightMatrix(layer_structure[i], layer_structure[i+1]);
        layer_updates[i] = CreateVector2D(NULL, layer_structure[i], layer_structure[i+1], false);
        bias_array[i] = CreateWeightMatrix(1, layer_structure[i+1]);
		
		
    }
}

/*
Xavier He initialization will be used...
*/
Vector2D * CreateWeightMatrix(int input_count, int output_count)
{
    float init_range = 0;
    Vector2D * temp = (Vector2D *)malloc(sizeof(Vector2D));
    Vector2D * device_temp;
    
    CHECK(cudaMalloc(&device_temp, sizeof(Vector2D)));
    
       
    temp->height = input_count; //For bias...
    temp->width = output_count;
    temp->data = (float * )malloc(sizeof(float)*(input_count)*output_count);
   
    
    

    init_range = sqrt(2.0 / input_count);

    for(int a=0; a<(input_count)*output_count; a++)
    {
        temp->data[a] = generate_uniform(-init_range, init_range);
    }
    
    float * temp2;
    CHECK(cudaMalloc(&temp2, sizeof(float)*temp->height*temp->width));

    CHECK(cudaMemcpy(temp2, temp->data, sizeof(float)*temp->height*temp->width, cudaMemcpyHostToDevice));
    
    
    CHECK(cudaMemcpy(&device_temp->data, &temp2, sizeof(float *), cudaMemcpyHostToDevice));
 
    CHECK(cudaMemcpy(&device_temp->height, &(temp->height), sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(&device_temp->width, &(temp->width), sizeof(int), cudaMemcpyHostToDevice));
    
    
    return device_temp;
}

Vector2D * device_ones = NULL, * device_transpose_ones;
int res_height;
Vector2D * device_input;
Vector2D * device_input_temp;
Vector2D * final_output;

Vector2D * device_input_transpose;
 dim3 grid;
void FeedForward(Vector2D * input, int input_width, int input_height)
{  
	int thread_block_x = 32, thread_block_y = 32;
	
	
	//printf("\n\n**************Feedforward*********** width : %d height : %d\n", input_width, input_height);

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
    
    if(device_ones == NULL)
    {
		
		float * ones_ = (float *)malloc(sizeof(float)*input_height);
    
		for(int i=0;i<input_height;i++)ones_[i] = 1.0;
		
		
		
		device_ones = CreateVector2D(ones_, 1, input_height, true);
		
		
		
		device_transpose_ones = CreateVector2D(ones_, input_height, 1, true);
		
		
		res_height = input_height;
		
		for(int current_layer = 1; current_layer < mlp_layer_count; current_layer++)
		{
			Vector2D * res = CreateVector2D(NULL, input_height, layer_structure[current_layer], false);
			layer_results[current_layer-1] = res;
			
			
			
				
			Vector2D * res2 = CreateVector2D(NULL, input_height, layer_structure[current_layer], false);
			bias_results[current_layer-1] = res2;
			

		}
		

		device_input_temp = input;
		device_input_transpose = CreateVector2D(NULL, input_width, input_height, false);
		device_input = device_input_temp;
		
	
		
	}
	else
	{
		
		device_input = device_input_temp;
		
	
		
	}
	
	//printf("\nInput data : \n");
	//DisplayVector2D<<<1, 1>>>(device_input);
	//cudaDeviceSynchronize();
	
	
	

    // By staring from first layer we are  performing forward pass iteratively...
    for(int current_layer=1; current_layer < mlp_layer_count; current_layer++)
    {
        {
             // weight matrix of the layer and the previous input is multiplied
             
            
             dim3 b(thread_block_x, thread_block_y);
			 dim3 grid( (layer_structure[current_layer]+b.x-1)/b.x, (input_height+b.y-1)/b.y);
             
             /*
            printf("\n\n\nLayer : %d\ndevice input : \n", current_layer-1);
            DisplayVector2D<<<1, 1>>>(device_input);
            cudaDeviceSynchronize();
            
            printf("\weight input : \n");
            DisplayVector2D<<<1, 1>>>(weight_array[current_layer-1]);
            cudaDeviceSynchronize();
            printf("\nresult vector .x %d .y %d\n", layer_structure[current_layer], res_height);
            printf("\ngrid.x : %d - grid.y : %d\n", grid.x, grid.y);
            
            */
            
            
             MatrixProduct<<<grid, b>>>(layer_results[current_layer-1], device_input, weight_array[current_layer-1]);
             cudaDeviceSynchronize();
            
             // Bias matrix is obtained...
             MatrixProduct<<<grid, b>>>(bias_results[current_layer-1], device_transpose_ones, bias_array[current_layer-1]);
             cudaDeviceSynchronize();
             
             // The bias is added to matmul operation...
             MatrixAdd<<<grid, b>>>(layer_results[current_layer-1], layer_results[current_layer-1], bias_results[current_layer-1]);
             cudaDeviceSynchronize();
            
             
             // If we are at output layer the hidden layer will be passed through the sigmoid function...
            if(current_layer < mlp_layer_count -1){
                Sigmoid<<<grid, b>>>(layer_results[current_layer-1], layer_results[current_layer-1]);
                //input = sigmoid(temp);
				 cudaDeviceSynchronize();
				}
            
            // If at output we are softmaxing the last hidden layer output
            else;
                //input = softmax(temp);
			
			
			device_input = layer_results[current_layer-1];
			
			

       }


     
    }
    final_output = device_input;
    
	

}



Vector2D * error;
Vector2D * label_data;
bool first_call = true;

Vector2D ** layer_weights_transpose;
Vector2D ** layer_results_transpose;
Vector2D * error_result;


Vector2D * batch_data;
Vector2D * batch_label;
Vector2D * whole_input_data;



void BackPropagate(Vector2D * input, Vector2D * labels, int input_width, int output_width, int batch_size)
{
	//printf("\n\nBackPropagation\n\n");
    // Firstly we are getting the outputs of each layer...
    int thread_block_x = 32, thread_block_y = 32;
    
    
		
	if(first_call == true)
	{
		first_call = false;
		
		
		layer_weights_transpose = (Vector2D **)malloc(sizeof(Vector2D*)*(mlp_layer_count-1));
		
		layer_results_transpose = (Vector2D **)malloc(sizeof(Vector2D*)*(mlp_layer_count-1));
		scalar_minus_array = (Vector2D **)malloc(sizeof(Vector2D*)*(mlp_layer_count-1));
		
		for(int a=0; a<mlp_layer_count-1;a++)
		{
			 
			layer_weights_transpose[a] = CreateVector2D(NULL, layer_structure[a+1], layer_structure[a], false);
			
		}
		
		 
		
		for(int current_layer = 1; current_layer < mlp_layer_count; current_layer++)
		{			
			Vector2D * res = CreateVector2D(NULL, layer_structure[current_layer], batch_size, false);
			layer_results_transpose[current_layer-1] = res;
			Vector2D * res2 = CreateVector2D(NULL, batch_size, layer_structure[current_layer], false);
			scalar_minus_array[current_layer-1] = res2;
		}
		 
		for(int a = 1; a<mlp_layer_count;a++)
		{
				layer_error_array[a-1] = CreateVector2D(NULL, batch_size, layer_structure[a],false);
				bias_updates[a-1] = CreateVector2D(NULL, 1, layer_structure[a], false);
		}
		error_result = CreateVector2D(NULL, batch_size, layer_structure[mlp_layer_count-1], false);
		 
		
	}
	

	
	FeedForward(input, input_width, batch_size);
	

    // We are calculating the error at outout....
    dim3 block(thread_block_x, thread_block_y);
    dim3 grid((output_width+block.x-1)/block.x, (batch_size+block.y-1)/block.y);
    
    
   
    
    MatrixSubtract<<<grid, block>>>(layer_error_array[mlp_layer_count-2], labels, final_output);
    cudaDeviceSynchronize();
  
	
   
   
    
    
    
    Vector2D * temp1;
    
     // We are starting from output layer...
    for( int current_layer = mlp_layer_count-2; current_layer>0;current_layer--)
    {
        // if we are output layer its weight should be adjust by simply
        // performing matrix multiplication with the previous layer output and output errpr...
        if(current_layer == mlp_layer_count-2)
        {
            // The previous layer's output is transposed...
            dim3 block(thread_block_x, thread_block_y);
            dim3 grid((layer_structure[current_layer]+block.x-1)/block.x, (batch_size+block.y-1)/block.y);
            
           
            TransposeVector2D<<<grid, block>>>(layer_results_transpose[current_layer-1], layer_results[current_layer-1]);
			
			cudaDeviceSynchronize();
			
			
            
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
            dim3 block2(layer_structure[mlp_layer_count-1] ,layer_structure[current_layer]);
            dim3 grid2((layer_structure[mlp_layer_count-1]+block.x-1)/block.x, (layer_structure[current_layer]+block.y-1)/block.y);
            
            
           
             MatrixProduct<<<grid2, block>>>(layer_updates[current_layer], layer_results_transpose[current_layer-1], layer_error_array[current_layer]);
             cudaDeviceSynchronize();
             
            
             
             dim3 temp(layer_structure[current_layer+1] ,1);
             dim3 grid3((layer_structure[current_layer+1]+block.x-1)/block.x, (1+block.y-1)/block.y);
             MatrixProduct<<<grid3, block>>>(bias_updates[current_layer], device_ones, layer_error_array[current_layer]);
			cudaDeviceSynchronize();
			
			
			
            
            continue;
        }

        
        temp1 = layer_error_array[current_layer+1] ;
       
        
        
      
        
        
        dim3 block3(layer_structure[current_layer+2], layer_structure[current_layer+1]);
        dim3 grid3((layer_structure[current_layer+2]+block.x-1)/block.x,  (layer_structure[current_layer+1]+block.y-1)/block.y);
        TransposeVector2D<<<grid3, block>>>(layer_weights_transpose[current_layer+1], weight_array[current_layer+1]);
		cudaDeviceSynchronize();
        
       
        
        
        
        // Error propagated to current layer is obtained by multiplying next layer's error with transpose of next layer's weight matrix
        // later pairwisely multiplying the output of next layer and with 1 - next layer's output....
       // error is propagated to next layer
       
       
       
        dim3 block4(layer_structure[current_layer+1], batch_size );
        dim3 grid4((layer_structure[current_layer+1]+block.x-1)/block.y, ( batch_size+block.y-1)/block.y);
        MatrixProduct<<<grid4, block>>>(layer_error_array[current_layer], temp1, layer_weights_transpose[current_layer+1]);
		cudaDeviceSynchronize();
		
		
        
	
		
		dim3 block5(layer_structure[current_layer+1], batch_size);
		dim3 grid5((layer_structure[current_layer+1]+block.x-1)/block.x, (batch_size+block.y-1)/block.y);
       ScalarMinusVector2D<<<grid5, block>>>(scalar_minus_array[current_layer], 1, layer_results[current_layer]);
       cudaDeviceSynchronize();
        
       
        
        
        // When we multiply prev error with the next layer's output
        /*
            next layer output was obtained via sigmoid, derivative of the sigmoid is (sigmoid*(1-sigmoid))
            While applying chaing rule this term takes places in the backprogation...
        */
        // So multiplying it we are exactly calculating the next layer's error...
       
        MatrixPairwiseProduct<<<grid5, block>>>(layer_error_array[current_layer], layer_error_array[current_layer], layer_results[current_layer]);
        
        cudaDeviceSynchronize();
        MatrixPairwiseProduct<<<grid5, block>>>(layer_error_array[current_layer], layer_error_array[current_layer], scalar_minus_array[current_layer]);
        cudaDeviceSynchronize();
		
		
		

        // We are transposing the prev layer output to calculate current layer's weight change...
        
        
        
        dim3 block6(layer_structure[current_layer], batch_size);
        dim3 grid6((layer_structure[current_layer]+block.x-1)/block.x, (batch_size+block.y-1)/block.y);
        TransposeVector2D<<<grid6, block>>>(layer_results_transpose[current_layer-1], layer_results[current_layer - 1 ]);
		cudaDeviceSynchronize();
        
         
        
        
        dim3 block7(layer_structure[current_layer+1], layer_structure[current_layer]);
        dim3 grid7((layer_structure[current_layer+1]+block.x-1)/block.x, (layer_structure[current_layer]+block.y-1)/block.y);
        
        
         MatrixProduct<<<grid7, block>>>(layer_updates[current_layer],  layer_results_transpose[current_layer-1], layer_error_array[current_layer]);
		cudaDeviceSynchronize();
        
        
        
        
        // Bias is also updated according to explanation above....
        dim3 block8(layer_structure[current_layer+1], 1);
        dim3 grid8((layer_structure[current_layer+1]+block.x-1)/block.x, (1+block.y-1)/block.y);
        MatrixProduct<<<grid8, block>>>( bias_updates[current_layer], device_ones, layer_error_array[current_layer]);
		cudaDeviceSynchronize();
		
		
        
        
		
		
        


        

    }


	
	
	
	 temp1 = layer_error_array[1];


	
       
        




	dim3 block9(layer_structure[2], layer_structure[1]);
    dim3 grid9((layer_structure[2]+block.x-1)/block.x, (layer_structure[1]+block.y-1)/block.y);
    TransposeVector2D<<<grid9, block>>>(layer_weights_transpose[1], weight_array[1]);
	cudaDeviceSynchronize();
	

	 
     
     dim3 block10(layer_structure[1], batch_size );
     dim3 grid10((layer_structure[1]+block.x-1)/block.x, (batch_size+block.y-1)/block.y);
     MatrixProduct<<<grid10, block>>>(layer_error_array[0], temp1, layer_weights_transpose[1]);
	 cudaDeviceSynchronize();
     
    
     
	dim3 block11(layer_structure[0+1], batch_size);
	dim3 grid11((layer_structure[0+1]+block.x-1)/block.x, (batch_size+block.y-1)/block.y);
       ScalarMinusVector2D<<<grid11, block>>>(scalar_minus_array[0], 1, layer_results[0]);
       cudaDeviceSynchronize();

     MatrixPairwiseProduct<<<grid11, block>>>(layer_error_array[0], layer_error_array[0], layer_results[0]);
     cudaDeviceSynchronize();
     MatrixPairwiseProduct<<<grid11, block>>>(layer_error_array[0], layer_error_array[0], scalar_minus_array[0]);
     cudaDeviceSynchronize();
		
	 
    



	 
	 dim3 block12(layer_structure[0], batch_size);
     dim3 grid12((layer_structure[0]+block.x-1)/block.x, (batch_size+block.y-1)/block.y);
     TransposeVector2D<<<grid12, block>>>(device_input_transpose , batch_data);
	 cudaDeviceSynchronize();
        
    
	
	dim3 block13(layer_structure[1], layer_structure[0]);
    dim3 grid13((layer_structure[1]+block.x-1)/block.x, (layer_structure[0]+block.y-1)/block.y);
    MatrixProduct<<<grid13, block>>>(layer_updates[0],  device_input_transpose, layer_error_array[0]);
	cudaDeviceSynchronize();


	 dim3 block14(layer_structure[0+1], 1);
	 dim3 grid14((layer_structure[0+1]+block.x-1)/block.x, (1+block.y-1)/block.y);
     MatrixProduct<<<grid14, block>>>( bias_updates[0], device_ones, layer_error_array[0]);
	cudaDeviceSynchronize();
		
	 
	

   


	
	
	for(int weight_index=0; weight_index < mlp_layer_count - 1;weight_index++)
    {
     

      
		dim3 grid((layer_structure[weight_index+1]+block.x-1)/block.x, (layer_structure[weight_index]+block.y-1)/block.y);
		ScalarMatrixProduct<<<grid, block>>>(layer_updates[weight_index], learning_rate, layer_updates[weight_index]);
        cudaDeviceSynchronize();
        MatrixAdd<<<grid, block>>>(weight_array[weight_index], weight_array[weight_index], layer_updates[weight_index]);
        cudaDeviceSynchronize();
        
        
        dim3 grid2((layer_structure[weight_index+1]+block.x-1)/block.x ,1);
        ScalarMatrixProduct<<<grid2, block>>>(bias_updates[weight_index], learning_rate, bias_updates[weight_index]);
        cudaDeviceSynchronize();
        MatrixAdd<<<grid2, block>>>(bias_array[weight_index], bias_array[weight_index], bias_updates[weight_index]);
        cudaDeviceSynchronize();
    }
    
    
  
	

}






Vector2D * CreateVector2DCPU(float * data, int height, int width)
{
    // A new structure is allocated in memory for matrix/vector...
    Vector2D * temp = (Vector2D *)malloc(sizeof(struct Vector2D));
    temp->data = data;
    temp->height = height;
    temp->width = width;
    return temp;
};




Vector2D * CreateOneHot(Vector2D * indexes, int vector_length)
{
    Vector2D * one_hot_vector = (Vector2D*)malloc(sizeof(Vector2D));
    one_hot_vector->height = indexes->height;
    one_hot_vector->width = vector_length;
    one_hot_vector->size = one_hot_vector->height;
    one_hot_vector->data = (float *)malloc(sizeof(float)*indexes->height*vector_length);
    memset(one_hot_vector->data, 0, sizeof(float)*indexes->height*vector_length);
    for(int i=0; i<one_hot_vector->height;i++)
    {
        one_hot_vector->data[i*vector_length+(int)indexes->data[i*indexes->width]] = 1.0;
    }
    return one_hot_vector;
}

#define kk 10






void DisplayVector2DCPU(Vector2D * vector)
{
    printf("[");
    for(int h = 0; h < vector->height; h++)
    {
        printf("[");
        for( int w = 0; w < vector->width-1; w++)
        {
            printf("%f, ", vector->data[h*vector->width+w]);
        }
        printf("%f], \n", vector->data[h*vector->width+vector->width-1]);
    }
    printf("\b\b\b]");
}
#define EMPTY printf("\n\n");



Vector2D * load_text_data()
{

    FILE * dosya = fopen("text_data.dat", "rb");

    int width, height;
    fread(&width, sizeof(int), 1, dosya);
    fread(&height, sizeof(int), 1, dosya);
     float * loaded_data = (float *)malloc(width*height*sizeof(float));
    for(int a =0; a< width*height; a++)
        fread(&loaded_data[a], sizeof(float), 1, dosya);


    fclose(dosya);

    printf("Width : %d - Height : %d\n", width, height);

   Vector2D * vec = CreateVector2DCPU(loaded_data, height, width);


  return vec;


}


Vector2D * load_label_data()
{

    FILE * dosya = fopen("label_data.dat", "rb");

    int width, height;
    fread(&width, sizeof(int), 1, dosya);
    fread(&height, sizeof(int), 1, dosya);
     float * loaded_data = (float *)malloc(width*height*sizeof(float));
    int value;
    for(int a =0; a< width*height; a++)
        {
        
        fread(&value, sizeof(int), 1, dosya);
		loaded_data[a] = value;
		}

    fclose(dosya);

    printf("Width : %d - Height : %d\n", width, height);

   Vector2D * vec = CreateVector2DCPU(loaded_data, height, width);
	

  return vec;


}






int main()
{
	srand(time(0));
	
	
	
	
	
	
	printf("MLP is being created....\n");
	CreateMLP(3, 32754, 160, 4);
	

	float error_val[32];

	
	
	int batch_size = 32;
	
	
	learning_rate = 0.0000001;

	#define ITERATION_COUNT 10
	double toplam= 0.0;
	
	
	Vector2D * data_set = load_text_data();
	Vector2D * labels_ = load_label_data();	
	printf("\nData loaded...\n");
	 Vector2D * one_hot_labels = CreateOneHot(labels_, 4);	
		
	
	
	
	label_data = CreateVector2D(one_hot_labels->data, one_hot_labels->height, one_hot_labels->width, true);
	
	batch_data = CreateVector2D(NULL, batch_size, data_set->width, false);
	whole_input_data = CreateVector2D(data_set->data, data_set->height, data_set->width, true); 
	
	batch_label = CreateVector2D(NULL, batch_size, one_hot_labels->width, false);

	
	
	
	for(int iteration = 1; iteration < ITERATION_COUNT; iteration++)
	{
		toplam = 0.0;
		
		for(int j = 0; j < data_set->height/batch_size; j++)
		{
			
		PointerSet<<<1, 1>>>(batch_data, whole_input_data, j*batch_size, batch_size);
		cudaDeviceSynchronize();
	
		PointerSet<<<1, 1>>>(batch_label, label_data, j*batch_size, batch_size);
		cudaDeviceSynchronize();
	
		
		FeedForward(batch_data, data_set->width, batch_size); 
		
		cudaDeviceSynchronize();
		
	
		dim3 block(32, 32);
		dim3 grids((one_hot_labels->width+block.x-1)/block.x, (batch_size+block.y-1)/block.y);
		
		MatrixSubtract<<<grids, block>>>(final_output, batch_label, final_output);
		cudaDeviceSynchronize();
		MatrixPairwiseProduct<<<grids, block>>>(final_output, final_output, final_output);
		cudaDeviceSynchronize();
		
		dim3 k(1, 32);
		Sum2D<<<1, k>>>(final_output);
		cudaDeviceSynchronize();
		cudaMemcpyFromSymbol( &error_val, error_sum, sizeof(float)*32);
	
		cudaDeviceSynchronize();
		for(int a=0;a<32;a++)
			toplam += error_val[a];
		}
		
		printf("\n\nIteration %d - Error : %f\n", iteration, toplam);
		
	
			for(int j = 0; j < data_set->height/batch_size; j++)
		{
			
		PointerSet<<<1, 1>>>(batch_data, whole_input_data, j*batch_size, batch_size);
		cudaDeviceSynchronize();
	
		PointerSet<<<1, 1>>>(batch_label, label_data, j*batch_size, batch_size);
		cudaDeviceSynchronize();
	
		
		
		BackPropagate(batch_data, batch_label, data_set->width, one_hot_labels->width, batch_size);
		cudaDeviceSynchronize();
		
	
		}
		
		
		
		
		
		
	}
	
	
	
	FeedForward(batch_data, data_set->width, batch_size);
		printf("\nFinal output : \n");
		DisplayVector2D<<<1, 1>>>(final_output);
		cudaDeviceSynchronize();

	
		printf("\nCorrect output : \n");
		DisplayVector2D<<<1, 1>>>(batch_label);
	
		cudaDeviceSynchronize();


    
	
	
	
	
	
	
	
	
	cudaDeviceReset();


}
