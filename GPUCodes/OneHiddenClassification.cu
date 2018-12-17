#include<stdio.h>

#include<stdio.h>
#include<string.h>
#include <stdlib.h>
#include <stdarg.h>
#include<time.h>
#include <math.h>
#include "MNIST_for_C-master/mnist.h"
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
		result->data[tid] = 1-vec1->data[tid];
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
__device__ double error_sum[32];
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
__global__ void ArgMax2D(Vector2D * result, Vector2D * vec1)
{
	
	int tid = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(tid ==0)
	{
		//printf("\nArgMax2Dvec->width : %d vec->height : %d - x dim %d y dim %d\n", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	
	if(blockDim.y*gridDim.y < vec1->height)
	{
		printf("\nArgMax2D\n");
		printf("vec->width : %d vec->height : %d - x dim %d y dim %d", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	} 
}
	
	if(tid < vec1->height)
	{
		float max = -100000;
		int max_index = 0;
		for(int a = 0; a < vec1->width;a++)
		{
			if(vec1->data[tid*vec1->width+a]>max)
			{
				max = vec1->data[tid*vec1->width+a];
				max_index = a;
			}
			
		}
		
		result->data[tid] = max_index;
		
		
	}
	
	
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


__global__ void Exponential(Vector2D * result, Vector2D * vec1)
{
	
	
	int tx = blockIdx.x*blockDim.x+ threadIdx.x;
	int ty = blockIdx.y*blockDim.y + threadIdx.y;
	int tid = ty*vec1->width+tx;
	if(tid ==0)
	{
		//printf("\nExponentialvec->width : %d vec->height : %d - x dim %d y dim %d\n", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	
	if((blockDim.y*gridDim.y < vec1->height) || (blockDim.x*gridDim.x<vec1->width))
	{
		printf("\Exponential\n");
		printf("vec->width : %d vec->height : %d - x dim %d y dim %d", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	} 
}
	
	if(tid < vec1->width*vec1->height)
	{
		result->data[tid] = exp(vec1->data[tid]);
	}
	
	
	
	
}










__global__ void Softmax(Vector2D * result, Vector2D * vec1)
{
	int tid = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(tid ==0)
	{
		//printf("\nSoftmaxvec->width : %d vec->height : %d - x dim %d y dim %d\n", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	
	if(blockDim.y*gridDim.y < vec1->height)
	{
		printf("\nSoftmax\n");
		printf("vec->width : %d vec->height : %d - x dim %d y dim %d", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	} 
}
	
	if(tid < vec1->height)
	{
		float toplam = 0;
		
		for(int a = 0; a < vec1->width;a++)
		{
			toplam += vec1->data[a+tid*vec1->width];
			
		}
		
		
		for(int a = 0; a < vec1->width;a++)
		{
			result->data[a+tid*vec1->width] = vec1->data[a+tid*vec1->width]/toplam;
			
		}
		
		
		
		
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




float generate_uniform(float a, float b)
{
return rand() / (RAND_MAX + 1.0) * (b - a) + a;
}


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

float learning_rate = 0.01;

Vector2D * w1, * w2, * b1, * b2;
Vector2D * output_1, * output_2; 
Vector2D * bias_result_1, * bias_result_2;



Vector2D * ones, * ones_transpose;


#define INPUT_NODE_COUNT 32754
#define HIDDEN_LAYER_NODE_COUNT 160
#define OUTPUT_NODE_COUNT 4



void FeedForward(Vector2D * device_input, int batch_size)
{
	//input * w1
	dim3 block(32, 32);
	dim3 grid((HIDDEN_LAYER_NODE_COUNT+block.x-1)/block.x, (batch_size+block.y-1)/block.y);
	MatrixProduct<<<grid, block>>>(output_1, device_input, w1);
	cudaDeviceSynchronize();
	
	
	
	
	//transpose ones * b1	
	MatrixProduct<<<grid, block>>>(bias_result_1, ones_transpose, b1);
	cudaDeviceSynchronize();

	
	
	
	
	
	
	
	//bias1 + input*w1
	MatrixAdd<<<grid, block>>>(output_1, output_1, bias_result_1);
	cudaDeviceSynchronize();
	
	
	// output of hidden layer...
	Sigmoid<<<grid, block>>>(output_1, output_1);
	cudaDeviceSynchronize();
	
	
	
		
	//output of hidden layer * w2
	grid.x = (OUTPUT_NODE_COUNT+block.x-1)/block.x; grid.y = (batch_size+block.y-1)/block.y;
	MatrixProduct<<<grid, block>>>(output_2, output_1, w2);
	cudaDeviceSynchronize();
	
	//transpose ones * b2
	MatrixProduct<<<grid, block>>>(bias_result_2, ones_transpose, b2);
	cudaDeviceSynchronize();
	//bias2 + output of hidden layer * w2 - final output....  
	MatrixAdd<<<grid, block>>>(output_2, output_2, bias_result_2);
	cudaDeviceSynchronize();
	
	
	Exponential<<<grid, block>>>(output_2, output_2);
	cudaDeviceSynchronize();
	grid.x = 1; block.x = 1;
	Softmax<<<grid, block>>>(output_2, output_2);
	cudaDeviceSynchronize();
		
}

Vector2D * layer_2_error, * layer_1_error;
Vector2D * w1_update, * w2_update, * b1_update, * b2_update; 
Vector2D * output_1_transpose, * input_transpose;
Vector2D * label_data;
Vector2D * device_whole_label;
Vector2D * device_whole_data;
Vector2D * device_input;
Vector2D * w2_transpose;
Vector2D * scalar_minus;

Vector2D * batch_data;
Vector2D * batch_label;


void BackPropagate(Vector2D * data, Vector2D * label, int batch_size)
{
	FeedForward(data, batch_size);
	
	
	//Output error calculation
	dim3 block(32, 32);
	dim3 grid((OUTPUT_NODE_COUNT+block.x-1)/block.x, (batch_size+block.y-1)/block.y);
	
	MatrixSubtract<<<grid, block>>>(layer_2_error, label, output_2);
	cudaDeviceSynchronize();
	
	
	//output1 transpose
	dim3 grid2((HIDDEN_LAYER_NODE_COUNT+block.x-1)/block.x, (batch_size+block.y-1)/block.y);
	TransposeVector2D<<<grid2, block>>>(output_1_transpose, output_1);
	cudaDeviceSynchronize();
	
	//W2 update...
	dim3 grid3((OUTPUT_NODE_COUNT+block.x-1)/block.x, (HIDDEN_LAYER_NODE_COUNT+block.y-1)/block.y);
	MatrixProduct<<<grid3, block>>>(w2_update, output_1_transpose, layer_2_error);
	cudaDeviceSynchronize();
	
	//b2 update
	dim3 grid4((OUTPUT_NODE_COUNT+block.x-1)/block.x, (1+block.y-1)/block.y);
	MatrixProduct<<<grid4, block>>>(b2_update, ones, layer_2_error);
	cudaDeviceSynchronize();
	
	//W2 transpose
	dim3 grid5((OUTPUT_NODE_COUNT+block.x-1)/block.x, (HIDDEN_LAYER_NODE_COUNT+block.y-1)/block.y);
	TransposeVector2D<<<grid5, block>>>(w2_transpose, w2);
	cudaDeviceSynchronize();	
	
	//Layer 1 error
	dim3 grid6((HIDDEN_LAYER_NODE_COUNT+block.x-1)/block.x, (batch_size+block.y-1)/block.y);
	MatrixProduct<<<grid6, block>>>(layer_1_error, layer_2_error, w2_transpose);
	cudaDeviceSynchronize();
	MatrixPairwiseProduct<<<grid6, block>>>(layer_1_error, layer_1_error, output_1);
	cudaDeviceSynchronize();
	
	ScalarMinusVector2D<<<grid6, block>>>(scalar_minus, 1.0, output_1);
	cudaDeviceSynchronize();
	
	MatrixPairwiseProduct<<<grid6, block>>>(layer_1_error, layer_1_error, scalar_minus);
	cudaDeviceSynchronize();
	
	//Input transpose
	dim3 grid7((INPUT_NODE_COUNT+block.x-1)/block.x, (batch_size+block.y-1)/block.y);
	TransposeVector2D<<<grid7, block>>>(input_transpose, data);
	cudaDeviceSynchronize();
	
	//w1 update....
	dim3 grid8((HIDDEN_LAYER_NODE_COUNT+block.x-1)/block.x, (INPUT_NODE_COUNT+block.y-1)/block.y);
	MatrixProduct<<<grid8, block>>>(w1_update, input_transpose, layer_1_error);
	cudaDeviceSynchronize();
	
	//b1 update
	dim3 grid9((HIDDEN_LAYER_NODE_COUNT+block.x-1)/block.x, (1+block.y-1)/block.y);
	MatrixProduct<<<grid9, block>>>(b1_update, ones, layer_1_error);
	
	//w2_update * learning rate
	dim3 grid10((OUTPUT_NODE_COUNT+block.x-1)/block.x, (HIDDEN_LAYER_NODE_COUNT+block.y-1)/block.y);
	ScalarMatrixProduct<<<grid10, block>>>(w2_update, learning_rate, w2_update);
	cudaDeviceSynchronize();
	
	//Apply w2 update
	MatrixAdd<<<grid10, block>>>(w2, w2, w2_update);
	cudaDeviceSynchronize();
	
	
	//b2_update * learning_rate
	dim3 grid11((OUTPUT_NODE_COUNT+block.x-1)/block.x, (1+block.y-1)/block.y);
	ScalarMatrixProduct<<<grid11, block>>>(b2_update, learning_rate, b2_update);
	cudaDeviceSynchronize();
	
	
	//Apply b2 update
	MatrixAdd<<<grid11, block>>>(b2, b2, b2_update);
	cudaDeviceSynchronize();
	
	//w1_update * leraning_rate
	dim3 grid12((HIDDEN_LAYER_NODE_COUNT+block.x-1)/block.x, (INPUT_NODE_COUNT+block.y-1)/block.y);
	ScalarMatrixProduct<<<grid12, block>>>(w1_update, learning_rate, w1_update);
	cudaDeviceSynchronize();
	
	//Apply w1 update
	MatrixAdd<<<grid12, block>>>(w1, w1, w1_update);
	cudaDeviceSynchronize();
	
	
	
	dim3 grid13((HIDDEN_LAYER_NODE_COUNT+block.x-1)/block.x, (1+block.y-1)/block.y);
	ScalarMatrixProduct<<<grid13, block>>>(b1_update, learning_rate, b1_update);
	cudaDeviceSynchronize();
	
	
	//Apply b1 update
	MatrixAdd<<<grid13, block>>>(b1, b1, b1_update);
	cudaDeviceSynchronize();
	
	
	
	
}


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
	int blockx = 32, blocky = 32;
	
	dim3 block(blockx, blocky);
	
	
	int batch_size = 32;
	
	float * ones_ = (float *)malloc(sizeof(float)*batch_size);
	for(int a = 0; a< batch_size;a++)ones_[a] = 1.0;
	ones = CreateVector2D(ones_, 1, batch_size, true);
	ones_transpose = CreateVector2D(ones_, batch_size, 1, true);
	
	
	//first hidden layer 160 input 784 
	w1 = CreateWeightMatrix(INPUT_NODE_COUNT, HIDDEN_LAYER_NODE_COUNT);
	b1 = CreateWeightMatrix(1, HIDDEN_LAYER_NODE_COUNT);
	bias_result_1 = CreateVector2D(NULL, batch_size, HIDDEN_LAYER_NODE_COUNT, false);
	
	output_1 = CreateVector2D(NULL, batch_size, HIDDEN_LAYER_NODE_COUNT, false);
	output_1_transpose = CreateVector2D(NULL, HIDDEN_LAYER_NODE_COUNT, batch_size, false);
	
	w1_update = CreateVector2D(NULL, INPUT_NODE_COUNT, HIDDEN_LAYER_NODE_COUNT, false);
	b1_update = CreateVector2D(NULL, 1, HIDDEN_LAYER_NODE_COUNT, false);
	
	//output 10 nodes....
	
	w2 = CreateWeightMatrix(HIDDEN_LAYER_NODE_COUNT, OUTPUT_NODE_COUNT);
	w2_transpose = CreateVector2D(NULL, OUTPUT_NODE_COUNT, HIDDEN_LAYER_NODE_COUNT, false);
	b2 = CreateWeightMatrix(1, OUTPUT_NODE_COUNT);
	bias_result_2 = CreateVector2D(NULL, batch_size, OUTPUT_NODE_COUNT, false);
	
	
	output_2 = CreateVector2D(NULL, batch_size, OUTPUT_NODE_COUNT, false);
	w2_update = CreateVector2D(NULL, HIDDEN_LAYER_NODE_COUNT, OUTPUT_NODE_COUNT, false);
	b2_update = CreateVector2D(NULL, 1, OUTPUT_NODE_COUNT, false);
	
	layer_2_error = CreateVector2D(NULL, batch_size, OUTPUT_NODE_COUNT, false);
	layer_1_error = CreateVector2D(NULL , batch_size, HIDDEN_LAYER_NODE_COUNT, false);
	
	scalar_minus  = CreateVector2D(NULL, batch_size, HIDDEN_LAYER_NODE_COUNT, false);
	
	input_transpose = CreateVector2D(NULL, INPUT_NODE_COUNT, batch_size, false);
	
	
					
	Vector2D * data_set = load_text_data();
	Vector2D * labels_ = load_label_data();	
	printf("\nData loaded...\n");
	 Vector2D * one_hot_labels = CreateOneHot(labels_, 4);	
	
	
	device_whole_data = CreateVector2D(data_set->data, data_set->height, 32754);
	device_whole_label = CreateVector2D(one_hot_labels->data, data_set->height, 4);
	
	batch_data = CreateVector2D(NULL, batch_size, INPUT_NODE_COUNT, false);
	batch_label = CreateVector2D(NULL, batch_size, OUTPUT_NODE_COUNT, false);
	
	
	
	double error_val[32];
	

	
		
	
	learning_rate = 0.0004;
	double toplam = 0;
	
	#define ITERATION_COUNT 40
	for(int iteration = 0 ; iteration < ITERATION_COUNT;iteration++)
	{
		toplam = 0;
		for(int batch_index = 1; batch_index < data_set->height/batch_size; batch_index++)
		{
	
			PointerSet<<<1 ,1>>>(batch_data, device_whole_data, (batch_index -1)*batch_size, batch_size);
			cudaDeviceSynchronize();
	
			PointerSet<<<1 ,1>>>(batch_label, device_whole_label, (batch_index-1)*batch_size, batch_size);
			cudaDeviceSynchronize();
	
			FeedForward(batch_data, batch_size);
	
			dim3 gridd((OUTPUT_NODE_COUNT+block.x-1)/block.x, (batch_size+block.y-1)/block.y);
			Log2D<<<gridd, block>>>(output_2, output_2);
			cudaDeviceSynchronize();
	
			MatrixPairwiseProduct<<<gridd, block>>>(layer_2_error, batch_label, output_2);
			cudaDeviceSynchronize();
			dim3 k(1, 32);
			Sum2D<<<1, k>>>(layer_2_error);
			cudaDeviceSynchronize();
			cudaMemcpyFromSymbol( &error_val, error_sum, sizeof(double)*32);
			//printf("\n\nIteration %d - Error : %f\n", iteration, error_val);	
			cudaDeviceSynchronize();
			for(int a=0;a<32;a++)
			toplam += -error_val[a];
		}
		printf("\nITeration %d NN error %f \n", iteration, toplam);
	
		for(int batch_index = 1; batch_index < data_set->height/batch_size; batch_index++)
		{
	
			PointerSet<<<1 ,1>>>(batch_data, device_whole_data, (batch_index -1)*batch_size, batch_size);
			cudaDeviceSynchronize();
	
			PointerSet<<<1 ,1>>>(batch_label, device_whole_label, (batch_index-1)*batch_size, batch_size);
			cudaDeviceSynchronize();
	
			BackPropagate(batch_data, batch_label, batch_size);
	
		}
	
	
	
	
	}
	
	
	
	FeedForward(batch_data, batch_size);
	printf("\nFinal output : \n");
	DisplayVector2D<<<1, 1>>>(output_2);
	
	cudaDeviceSynchronize();

	
	printf("\nCorrect output : \n");
	DisplayVector2D<<<1, 1>>>(batch_label);
	
	cudaDeviceSynchronize();

	
	
	printf("\nTamam\n");
	cudaDeviceReset();

}
