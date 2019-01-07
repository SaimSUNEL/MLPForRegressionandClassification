#include<stdio.h>

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
#define ITERATION_COUNT 300

#define BATCH_SIZE 256
#define BLOCK_X 32
#define BLOCK_Y 32
float learning_rate = 1.0e-7;

#define INPUT_NODE_COUNT 32754
#define HIDDEN_LAYER_NODE_COUNT 128
#define OUTPUT_NODE_COUNT 32

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




__global__ void MatrixSubtract(Vector2D * __restrict__ result, Vector2D * __restrict__ vec1, Vector2D * __restrict__ vec2)
{
	
		if((vec1->width != vec2->width) ||   (vec1->height != vec2->height))
	{
		printf("\n\n**********Matrix Subtract diff dimension....");
		return;
	}
	
	
	int tx = blockIdx.x*blockDim.x*4+ threadIdx.x;
	int ty = blockIdx.y*blockDim.y + threadIdx.y;
	int tid = ty*vec1->width+tx;
	
	if(tid ==0)
	{
		//printf("\nMatrixSubtractvec->width : %d vec->height : %d - x dim %d y dim %d\n", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	
	if((blockDim.y*gridDim.y < vec1->height) || (blockDim.x*gridDim.x<vec1->width/4))
	{
		printf("\nMatrixSubtract\n");
		printf("vec->width : %d vec->height : %d - x dim %d y dim %d", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	} 
}
	
	if(tid +3*blockDim.x< vec1->width*vec1->height)
	{
		result->data[tid] = vec1->data[tid] - vec2->data[tid];
		result->data[tid+blockDim.x] = vec1->data[tid+blockDim.x] - vec2->data[tid+blockDim.x];
		result->data[tid+2*blockDim.x] = vec1->data[tid+2*blockDim.x] - vec2->data[tid+2*blockDim.x];
		result->data[tid+3*blockDim.x] = vec1->data[tid+3*blockDim.x] - vec2->data[tid+3*blockDim.x];
		
	}
	
	
}




#define TILE_WIDTH BLOCK_X
#define TILE_HEIGHT BLOCK_Y
__global__ void MatrixProductShared(  Vector2D * __restrict__ result, Vector2D * __restrict__ m1, Vector2D * __restrict__ m2 )//float *A, float *B, float *C ) {
 {
  __shared__ float A_tile[TILE_HEIGHT][TILE_WIDTH];
  
  __shared__ float B_tile[TILE_HEIGHT][TILE_WIDTH+1];

   int numARows = m1->height, numAColumns= m1->width, numBRows = m2->height, numBColumns = m2->width, numCRows = result->height, numCColumns = m2->width;
    
   float * A = m1->data, * B = m2->data, * C = result->data; 
    
  float sum = 0.0;

  // where am I?
  // tx for thread_x or tile_x
  int tx = threadIdx.x; int ty = threadIdx.y;
  // cx for top left corner of tile in C
  int cx = blockIdx.x * blockDim.x; int cy = blockIdx.y * blockDim.y;
  // Cx for cell coordinates in C
  int Cx = cx + tx; int Cy = cy + ty;

  int total_tiles = (numAColumns + TILE_WIDTH - 1) / TILE_WIDTH;

  for (int tile_idx = 0; tile_idx < total_tiles; tile_idx++) {
    // the corresponding tiles' top left corners are:
    // for A: row = blockIdx.y * blockDim.y, col = tile_idx * TILE_WIDTH
    // for B: row = tile_idx * TILE_WIDTH, col = blockIdx.x * blockDim.x
    
    // loading tiles
    int Ax = tile_idx * TILE_WIDTH + tx; int Ay = cy + ty;
    int Bx = cx + tx; int By = tile_idx * TILE_WIDTH + ty;

    if (Ax < numAColumns && Ay < numARows) {
      A_tile[ty][tx] = A[Ay * numAColumns + Ax];
    }
    else {
      A_tile[ty][tx] = 0.0;
    }
    if (Bx < numBColumns && By < numBRows) {
      B_tile[ty][tx] = B[By * numBColumns + Bx];
    }
    else {
      B_tile[ty][tx] = 0.0;
    }
    __syncthreads();

    // multiplying tiles
    #pragma unroll 4
    for (int i = 0; i < TILE_WIDTH; i++) {
      sum += A_tile[ty][i] * B_tile[i][tx];
    }
    __syncthreads();
  }

  // saving result (discarded if we're in the wrong thread)
  if (Cx < numCColumns && Cy < numCRows) {
    C[Cy * numCColumns + Cx] = sum;
  }
}



__global__ void TransposeVector2DShared(Vector2D * __restrict__ res, Vector2D * __restrict__ m1)
{
	
	int thx = blockIdx.x*blockDim.x+ threadIdx.x;
	int thy = blockIdx.y*blockDim.y+threadIdx.y;
	
	int tid = thx + thy*m1->width;
	
	 
	 __shared__ float ordered_data[BLOCK_Y][BLOCK_X+1];
	 __shared__ float transposed_data[BLOCK_Y][BLOCK_X+1];

	 

	
	

	
	
	
	int j = threadIdx.x+blockDim.x*blockIdx.y;
	
	int k = threadIdx.y + blockDim.y*blockIdx.x;
	
	
	
	
	int target = j + res->width*k;
	
	
	
	
	
	if(tid < m1->width*m1->height)
	{
	//padded
	ordered_data[threadIdx.y][threadIdx.x] = m1->data[tid] ;
	
	}

	__syncthreads();
	
	
	//transposed_data[thy+thx*m1->height] = ordered_data[tid] ;
	if(thx < m1->width && thy< m1->height)
	{
		
	transposed_data[threadIdx.x][threadIdx.y]  = ordered_data[threadIdx.y][threadIdx.x];

	}
__syncthreads();

if(thx < m1->width && thy< m1->height)
	{
	res->data [target] = transposed_data[threadIdx.y][threadIdx.x] ;
	
	
	//printf("idy : %d - idx : %d - blockdim x : %d - blockDim y : %d - gridDim.x - %d - gridDim.y : %d\n", thy, thx, blockDim.x, blockDim.y, gridDim.x, gridDim.y);
	}
	
	
	
	
	
	
	if(tid ==0)
	{
		//printf("\nTransposeVector2Dvec->width : %d vec->height : %d - x dim %d y dim %d\n", m1->width, m1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	
	if((blockDim.y*gridDim.y < m1->height) || (blockDim.x*gridDim.x<m1->width))
	{
		printf("\nTransposeVector2D\n");
		printf("vec->width : %d vec->height : %d - x dim %d y dim %d", m1->width, m1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	} 
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







__device__ float error_sum[BATCH_SIZE];
__global__ void Sum2D(Vector2D * __restrict__ vec)
{
	int tid = threadIdx.y;
	int val = 0;
	int width = vec->width;
	#pragma unroll 4
	for(int a = 0; a < width; a++)
	{
		val += vec->data[a+tid*width];
	}	
	error_sum[tid] = val;
}

__device__ int arg_max_result[BATCH_SIZE];

__global__ void ArgMax2D(Vector2D * __restrict__ vec1)
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
		#pragma unroll 4
		for(int a = 0; a < vec1->width;a++)
		{
			if(vec1->data[tid*vec1->width+a]>max)
			{
				max = vec1->data[tid*vec1->width+a];
				max_index = a;
			}
			
		}
		
		arg_max_result[tid] = max_index;
		
		
	}
	
	
}









__global__ void Softmax(Vector2D * __restrict__ result, Vector2D * __restrict__ vec1)
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
		#pragma unroll 4
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

__global__ void AddandSigmoid(Vector2D * __restrict__ result, Vector2D * __restrict__ vec1, Vector2D * __restrict__ vec2)
{
	if((vec1->width != vec2->width) ||   (vec1->height != vec2->height))
	{
		printf("\n\n**********Matrix add diff dimension....");
		return;
	}
	
	
	int tx = blockIdx.x*blockDim.x*4+ threadIdx.x;
	int ty = blockIdx.y*blockDim.y + threadIdx.y;
	int tid = ty*vec1->width+tx;
	if(tid ==0)
	{//printf("\nMatrixAddvec->width : %d vec->height : %d - x dim %d y dim %d\n", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	
	if((blockDim.y*gridDim.y < vec1->height) || (blockDim.x*gridDim.x<vec1->width/4))
	{
		printf("\AddandSigmoid\n");
		printf("vec->width : %d vec->height : %d - x dim %d y dim %d", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	} 
}
	
	
	if(tid + blockDim.x*3 < vec1->width*vec1->height)
	{
		result->data[tid] = 1.0/(1+exp(-(vec1->data[tid] + vec2->data[tid])));
		result->data[tid+blockDim.x] = 1.0/(1+exp(-(vec1->data[tid+blockDim.x] + vec2->data[tid+blockDim.x])));
		result->data[tid+2*blockDim.x] = 1.0/(1+exp(-(vec1->data[tid+2*blockDim.x] + vec2->data[tid+2*blockDim.x])));
		result->data[tid+3*blockDim.x] = 1.0/(1+exp(-(vec1->data[tid+3*blockDim.x] + vec2->data[tid+3*blockDim.x])));
	}
	
	
}


__global__ void AddandExponential(Vector2D * __restrict__ result, Vector2D * __restrict__ vec1, Vector2D * __restrict__ vec2)
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
		printf("\AddandExponential\n");
		printf("vec->width : %d vec->height : %d - x dim %d y dim %d", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	} 
}
	
	if(tid  < vec1->width*vec1->height)
	{
		result->data[tid] = exp(vec1->data[tid] + vec2->data[tid]);
		
	
	
	
	}
	
	
}



//Combination of matrixpairwise-Scalarminus-matrixpairwise in backpropagte....
__global__ void LayerErrorCalculate(Vector2D * __restrict__ result, Vector2D * __restrict__ vec1, Vector2D * __restrict__ vec2)
{
	
	
		if((vec1->width != vec2->width) ||   (vec1->height != vec2->height))
	{
		printf("\n\n**********MatrixPairwiseProduct dimension....");
		return;
	}
	
	
	
	int tx = blockIdx.x*blockDim.x*4+ threadIdx.x;
	int ty = blockIdx.y*blockDim.y + threadIdx.y;
	int tid = ty*vec1->width+tx;
	
	
	if(tid ==0)
	{
		//printf("\nMatrixPairwiseProductvec->width : %d vec->height : %d - x dim %d y dim %d\n", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	
	if((blockDim.y*gridDim.y < vec1->height) || (blockDim.x*gridDim.x<vec1->width/4))
	{
		printf("\nMatrixPairwiseProduct\n");
		printf("vec->width : %d vec->height : %d - x dim %d y dim %d", vec1->width, vec2->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	} 
}
	
	
	if(tid + 3*blockDim.x< vec1->width*vec1->height)
	{
		result->data[tid] = vec1->data[tid] * vec2->data[tid]*(1-vec2->data[tid]) ;
		result->data[tid+blockDim.x] = vec1->data[tid+blockDim.x] * vec2->data[tid+blockDim.x]*(1-vec2->data[tid+blockDim.x]) ;
		result->data[tid+2*blockDim.x] = vec1->data[tid+2*blockDim.x] * vec2->data[tid+2*blockDim.x]*(1-vec2->data[tid+2*blockDim.x]) ;
		result->data[tid+3*blockDim.x] = vec1->data[tid+3*blockDim.x] * vec2->data[tid+3*blockDim.x]*(1-vec2->data[tid+3*blockDim.x]) ;
	}
	
	
}

__global__ void ApplyWeightChange(Vector2D * __restrict__ result, float learning_rate, Vector2D * __restrict__ source)
{
	if((result->width != source->width) ||   (result->height != source->height))
	{
		printf("\n\n**********ScalarMatrixProduct dimensionç....");
		return;
	}
	int tx = blockIdx.x*blockDim.x*4+ threadIdx.x;
	int ty = blockIdx.y*blockDim.y + threadIdx.y;
	int tid = ty*source->width+tx;
	
	if(tid ==0)
	{
		//printf("\nScalarMatrixProductvec->width : %d vec->height : %d - x dim %d y dim %d\n", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	
	if((blockDim.y*gridDim.y < source->height) || (blockDim.x*gridDim.x<source->width/4))
	{
		printf("\nScalarMatrixProduct\n");
		printf("vec->width : %d vec->height : %d - x dim %d y dim %d", source->width, source->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	} 
}
	
	if(tid +3*blockDim.x< source->width*source->height)
	{
		result->data[tid] += learning_rate*source->data[tid];
		result->data[tid+blockDim.x] += learning_rate*source->data[tid+blockDim.x];
		result->data[tid+2*blockDim.x] += learning_rate*source->data[tid+2*blockDim.x];
		result->data[tid+3*blockDim.x] += learning_rate*source->data[tid+3*blockDim.x];
		
	}
	
	
}





__global__ void Vector2DInfo(Vector2D * vec)
{
	printf("\n\nWidth : %d - height : %d\n\n", vec->width, vec->height);
	
	
	
	
}



__global__ void calculateCrossEntropyLoss(Vector2D * __restrict__ result, Vector2D * __restrict__ vec1, Vector2D * __restrict__ vec2)
{
	
	
		if((vec1->width != vec2->width) ||   (vec1->height != vec2->height))
	{
		printf("\n\n**********MatrixPairwiseProduct dimension....");
		return;
	}
	
	
	
	int tx = blockIdx.x*blockDim.x*4+ threadIdx.x;
	int ty = blockIdx.y*blockDim.y + threadIdx.y;
	int tid = ty*vec1->width+tx;
	
	
	if(tid ==0)
	{
		//printf("\nMatrixPairwiseProductvec->width : %d vec->height : %d - x dim %d y dim %d\n", vec1->width, vec1->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	
	if((blockDim.y*gridDim.y < vec1->height) || (blockDim.x*gridDim.x<vec1->width/4))
	{
		printf("\nMatrixPairwiseProduct\n");
		printf("vec->width : %d vec->height : %d - x dim %d y dim %d", vec1->width, vec2->height, blockDim.x*gridDim.x, blockDim.y*gridDim.y);
	} 
}
	
	
	if(tid +3*blockDim.x< vec1->width*vec1->height)
	{
		result->data[tid] = vec1->data[tid] * log(vec2->data[tid]);
		result->data[tid+blockDim.x] = vec1->data[tid+blockDim.x] * log(vec2->data[tid+blockDim.x]);
		result->data[tid+2*blockDim.x] = vec1->data[tid+2*blockDim.x] * log(vec2->data[tid+2*blockDim.x]);
		result->data[tid+3*blockDim.x] = vec1->data[tid+3*blockDim.x] * log(vec2->data[tid+3*blockDim.x]);
		
	}
	
	
}








#define EMPTY printf("\n\n");



Vector2D * w1, * w2, * b1, * b2;
Vector2D * output_1, * output_2; 
Vector2D * bias_result_1, * bias_result_2;



Vector2D * ones, * ones_transpose;





void FeedForward(Vector2D * device_input, int batch_size)
{
	//input * w1
	dim3 block(BLOCK_X, BLOCK_Y);
	dim3 grid((HIDDEN_LAYER_NODE_COUNT+block.x-1)/block.x, (batch_size+block.y-1)/block.y);
	MatrixProductShared<<<grid, block>>>(output_1, device_input, w1);
	cudaDeviceSynchronize();
	
	
	
	
	//transpose ones * b1	
	MatrixProductShared<<<grid, block>>>(bias_result_1, ones_transpose, b1);
	cudaDeviceSynchronize();

	
	
	int temp = grid.x ;
	
	
	grid.x /=4;
	if(grid.x == 0)grid.x = 1;
	AddandSigmoid<<<grid, block>>>(output_1, output_1, bias_result_1);
	cudaDeviceSynchronize();
	grid.x =temp;
	
	/*
	//bias1 + input*w1
	MatrixAdd<<<grid, block>>>(output_1, output_1, bias_result_1);
	cudaDeviceSynchronize();
	
	
	// output of hidden layer...
	Sigmoid<<<grid, block>>>(output_1, output_1);
	cudaDeviceSynchronize();
	*/
	
	
		
	//output of hidden layer * w2
	grid.x = (OUTPUT_NODE_COUNT+block.x-1)/block.x; grid.y = (batch_size+block.y-1)/block.y;
	MatrixProductShared<<<grid, block>>>(output_2, output_1, w2);
	cudaDeviceSynchronize();
	
	
	
	
	
	//transpose ones * b2
	MatrixProductShared<<<grid, block>>>(bias_result_2, ones_transpose, b2);
	cudaDeviceSynchronize();
	
	
	
	
	
	
	
	
	
	
	
	AddandExponential<<<grid, block>>>(output_2, output_2, bias_result_2);
	
	
	cudaDeviceSynchronize();
	
	
	
	
	
	
	
	
	
	
	/*
	//bias2 + output of hidden layer * w2 - final output....  
	MatrixAdd<<<grid, block>>>(output_2, output_2, bias_result_2);
	cudaDeviceSynchronize();
	
	
	Exponential<<<grid, block>>>(output_2, output_2);
	cudaDeviceSynchronize();
	*/
	
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
	int temp;
	
	//Output error calculation
	dim3 block(BLOCK_X, BLOCK_Y);
	dim3 grid((OUTPUT_NODE_COUNT+block.x-1)/block.x, (batch_size+block.y-1)/block.y);
	temp = grid.x ;
	grid.x /= 4;
	if(grid.x ==0 )grid.x = 1;
	
	MatrixSubtract<<<grid, block>>>(layer_2_error, label, output_2);
	grid.x = temp;
	
	
	
	//output1 transpose
	dim3 grid2((HIDDEN_LAYER_NODE_COUNT+block.x-1)/block.x, (batch_size+block.y-1)/block.y);
	TransposeVector2DShared<<<grid2, block>>>(output_1_transpose, output_1);
	cudaDeviceSynchronize();

	
	
	//W2 update...
	dim3 grid3((OUTPUT_NODE_COUNT+block.x-1)/block.x, (HIDDEN_LAYER_NODE_COUNT+block.y-1)/block.y);
	MatrixProductShared<<<grid3, block>>>(w2_update, output_1_transpose, layer_2_error);
	
	
	//b2 update
	dim3 grid4((OUTPUT_NODE_COUNT+block.x-1)/block.x, (1+block.y-1)/block.y);
	MatrixProductShared<<<grid4, block>>>(b2_update, ones, layer_2_error);
	

	//W2 transpose
	dim3 grid5((OUTPUT_NODE_COUNT+block.x-1)/block.x, (HIDDEN_LAYER_NODE_COUNT+block.y-1)/block.y);
	TransposeVector2DShared<<<grid5, block>>>(w2_transpose, w2);
	cudaDeviceSynchronize();	
	
	
	
	//Layer 1 error
	dim3 grid6((HIDDEN_LAYER_NODE_COUNT+block.x-1)/block.x, (batch_size+block.y-1)/block.y);
	MatrixProductShared<<<grid6, block>>>(layer_1_error, layer_2_error, w2_transpose);
	cudaDeviceSynchronize();
	temp = grid.x;
	grid.x /=4;
	if(grid.x == 0)grid.x = 1;
	
	LayerErrorCalculate<<<grid6, block>>>(layer_1_error, layer_1_error, output_1);
	grid.x =temp;
	
	/*
	MatrixPairwiseProduct<<<grid6, block>>>(layer_1_error, layer_1_error, output_1);
	cudaDeviceSynchronize();
	
	ScalarMinusVector2D<<<grid6, block>>>(scalar_minus, 1.0, output_1);
	cudaDeviceSynchronize();
	
	MatrixPairwiseProduct<<<grid6, block>>>(layer_1_error, layer_1_error, scalar_minus);
	cudaDeviceSynchronize();
	*/
	
	//Input transpose
	dim3 grid7((INPUT_NODE_COUNT+block.x-1)/block.x, (batch_size+block.y-1)/block.y);
	TransposeVector2DShared<<<grid7, block>>>(input_transpose, data);
	cudaDeviceSynchronize();
	
	
	
	
	//w1 update....
	dim3 grid8((HIDDEN_LAYER_NODE_COUNT+block.x-1)/block.x, (INPUT_NODE_COUNT+block.y-1)/block.y);
	MatrixProductShared<<<grid8, block>>>(w1_update, input_transpose, layer_1_error);
	
	//b1 update
	dim3 grid9((HIDDEN_LAYER_NODE_COUNT+block.x-1)/block.x, (1+block.y-1)/block.y);
	MatrixProductShared<<<grid9, block>>>(b1_update, ones, layer_1_error);
	
	
	
	cudaDeviceSynchronize();
	//Burası
	
	
	
	//w2_update * learning rate
	dim3 grid10((OUTPUT_NODE_COUNT+block.x-1)/block.x, (HIDDEN_LAYER_NODE_COUNT+block.y-1)/block.y);
		
		temp = grid10.x ;
		grid10.x /= 4;
		if(grid10.x ==0)grid10.x = 1; 
		ApplyWeightChange<<<grid10, block>>>(w2, learning_rate, w2_update);
		grid10.x = temp;
	
	
	/*ScalarMatrixProduct<<<grid10, block>>>(w2_update, learning_rate, w2_update);
	cudaDeviceSynchronize();
	
	//Apply w2 update
	MatrixAdd<<<grid10, block>>>(w2, w2, w2_update);
	cudaDeviceSynchronize();
	*/

	
	
	//b2_update * learning_rate
	
	
	dim3 grid11((OUTPUT_NODE_COUNT+block.x-1)/block.x, (1+block.y-1)/block.y);
		temp = grid11.x ;
		grid11.x /= 4;
		if(grid11.x == 0) grid11.x = 1;
		ApplyWeightChange<<<grid11, block>>>(b2, learning_rate, b2_update);
		grid11.x = temp;
	/*ScalarMatrixProduct<<<grid11, block>>>(b2_update, learning_rate, b2_update);
	cudaDeviceSynchronize();
	
	
	//Apply b2 update
	MatrixAdd<<<grid11, block>>>(b2, b2, b2_update);
	cudaDeviceSynchronize();
	*/
	
	
	
	//w1_update * leraning_rate
	dim3 grid12((HIDDEN_LAYER_NODE_COUNT+block.x-1)/block.x, (INPUT_NODE_COUNT+block.y-1)/block.y);
		temp = grid12.x;
		grid12.x /= 4;
		if(grid12.x == 0)grid12.x = 1;
		ApplyWeightChange<<<grid12, block>>>(w1, learning_rate, w1_update);

	/*
	ScalarMatrixProduct<<<grid12, block>>>(w1_update, learning_rate, w1_update);
	cudaDeviceSynchronize();
	
	//Apply w1 update
	MatrixAdd<<<grid12, block>>>(w1, w1, w1_update);
	cudaDeviceSynchronize();
	*/
	
	
	dim3 grid13((HIDDEN_LAYER_NODE_COUNT+block.x-1)/block.x, (1+block.y-1)/block.y);
		temp = grid13.x;
		grid13.x /= 4;
		if(grid13.x == 0)grid13.x = 1;
		ApplyWeightChange<<<grid13, block>>>(b1, learning_rate, b1_update);

	/*
	ScalarMatrixProduct<<<grid13, block>>>(b1_update, learning_rate, b1_update);
	cudaDeviceSynchronize();
	
	
	//Apply b1 update
	MatrixAdd<<<grid13, block>>>(b1, b1, b1_update);
	cudaDeviceSynchronize();
	*/
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


Vector2D * load_test_text_data()
{

    FILE * dosya = fopen("test_text_data.dat", "rb");

    int width, height;
    fread(&width, sizeof(int), 1, dosya);
    fread(&height, sizeof(int), 1, dosya);
     float * loaded_data = (float*)malloc(width*height*sizeof(float));
    for(int a =0; a< width*height; a++)
        fread(&loaded_data[a], sizeof(float), 1, dosya);


    fclose(dosya);

    printf("Width : %d - Height : %d\n", width, height);

   Vector2D * vec = CreateVector2DCPU(loaded_data, height, width);


  return vec;


}


Vector2D * load_test_label_data()
{

    FILE * dosya = fopen("test_label_data.dat", "rb");

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

Vector2D * device_whole_test_data, * device_whole_test_label_data;

int main()
{
	/*
	float * del  = (float *)malloc(sizeof(float)*64*32);
	for(int a=0; a < 64*32 ; a++)del[a] = a+1;
	
	Vector2D * s = CreateVector2D(del, 32, 64, true);
	Vector2D * transposed = CreateVector2D(NULL, 64, 32, false);
	
	dim3 bl(32, 32);
	dim3 gri((64+bl.x-1)/bl.x, (32+bl.y-1)/bl.y);
	TransposeVector2DShared<<<gri, bl>>>(transposed, s);
	cudaDeviceSynchronize();
	printf("\nOriginal matrix : \n");
	DisplayVector2D<<<1, 1>>>(s);
	cudaDeviceSynchronize();
	
	
	printf("\nTransposed matrix \n");
	DisplayVector2D<<<1, 1>>>(transposed);
	cudaDeviceSynchronize();
	
	
	
	exit(0);
	*/
	
	int count = 0;
	cudaGetDeviceCount(&count);

	
    clock_t train_start, train_end;
	
	clock_t execution_start, execution_end;
	
	clock_t program_start, program_end ;
	program_start = clock();
	
	execution_start = clock();
	
	srand(time(0));
	int blockx = 32, blocky = 32;
	
	dim3 block(blockx, blocky);
	
	
	int batch_size = BATCH_SIZE;
	
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
	Vector2D * one_hot_labels = CreateOneHot(labels_, OUTPUT_NODE_COUNT);	
	
	
	Vector2D * test_data = load_test_text_data();
    Vector2D * lab = load_test_label_data();
    Vector2D *one_hot_test = CreateOneHot(lab, OUTPUT_NODE_COUNT);

	
	
	
	device_whole_data = CreateVector2D(data_set->data, data_set->height, 32754);
	device_whole_label = CreateVector2D(one_hot_labels->data, data_set->height, OUTPUT_NODE_COUNT);
	
	batch_data = CreateVector2D(NULL, batch_size, INPUT_NODE_COUNT, false);
	batch_label = CreateVector2D(NULL, batch_size, OUTPUT_NODE_COUNT, false);
	
	device_whole_test_data = CreateVector2D(test_data->data, test_data->height, 32754);
	device_whole_test_label_data = CreateVector2D(one_hot_test->data, one_hot_test->height, OUTPUT_NODE_COUNT);
	
	
	
	float error_val[BATCH_SIZE];
	float batch_error;
	double toplam = 0;
	double previous_error = 100000000000;
	
	for(int iteration = 0 ; iteration < ITERATION_COUNT;iteration++)
	{
		toplam = 0;
		
		
	
		train_start = clock();
	int temp;
		
		for(int batch_index = 1; batch_index <data_set->height/batch_size; batch_index++)//10;batch_index++ );//data_set->height/batch_size; batch_index++)
		{
			
			PointerSet<<<1 ,1>>>(batch_data, device_whole_data, (batch_index -1)*batch_size, batch_size);
			cudaDeviceSynchronize();
	
			PointerSet<<<1 ,1>>>(batch_label, device_whole_label, (batch_index-1)*batch_size, batch_size);
			cudaDeviceSynchronize();
	
			BackPropagate(batch_data, batch_label, batch_size);
			
			
			
			dim3 gridd((OUTPUT_NODE_COUNT+block.x-1)/block.x, (batch_size+block.y-1)/block.y);
			

			temp = gridd.x ;
			gridd.x /= 4;
			if(gridd.x == 0)gridd.x = 1;
			
			calculateCrossEntropyLoss<<<gridd, block>>>(layer_2_error, batch_label, output_2);
			
			gridd.x = temp;
			
			cudaDeviceSynchronize();
			dim3 k(1, BATCH_SIZE);
			Sum2D<<<1, k>>>(layer_2_error);
			cudaDeviceSynchronize();
			cudaMemcpyFromSymbol( &error_val, error_sum, sizeof(float)*BATCH_SIZE);
			//printf("\n\nIteration %d - Error : %f\n", iteration, error_val);	
			cudaDeviceSynchronize();
			for(int a=0;a<BATCH_SIZE;a++)
			toplam += -error_val[a];
			
	
		}
		printf("\nITeration %d error %f \n", iteration, toplam);
		/*
		if(toplam < previous_error)
		learning_rate += 2.0e-10;
		else
		learning_rate -= 2.0e-10;
		previous_error = toplam;
		*/
		if(previous_error > 1000000) previous_error = toplam;
		learning_rate = learning_rate + 0.00000001*(previous_error - toplam);
		
		previous_error = toplam;
		train_end = clock();
        printf("\nIteration %d - Whole data train time : %f\n\n", iteration, (double)(train_end - train_start) / CLOCKS_PER_SEC);


	
	
	
	}
	
	if(abs(previous_error - toplam) < 0.00001)
	{
     learning_rate -= learning_rate/10.0*2;
     
     }
	
	printf("\nTraining has finished...\n");
	execution_end = clock();
	printf("\nWhole data train time : %f\n\n", (double)(execution_end - execution_start) / CLOCKS_PER_SEC);

	
	
	Vector2D * batch_test_data, * batch_label_data;
	batch_test_data = CreateVector2D(NULL, batch_size, test_data->width, false);
	batch_label_data = CreateVector2D(NULL, batch_size, one_hot_test->width, false);
	
	int predicted_labels[BATCH_SIZE];
	
	int correct_number = 0, false_number = 0;
	
	
	for(int batch_index = 0; batch_index < one_hot_test->height/batch_size; batch_index++)//10;batch_index++)//one_hot_test->height/batch_size; batch_index++)
	{

	PointerSet<<<1 ,1>>>(batch_test_data, device_whole_test_data, (batch_index)*batch_size, batch_size);
	cudaDeviceSynchronize();
	
	PointerSet<<<1 ,1>>>(batch_label_data, device_whole_test_label_data, (batch_index)*batch_size, batch_size);
	cudaDeviceSynchronize();
	//printf("\nFeed forward...\n");
	FeedForward(batch_test_data, batch_size);
	//printf("\nArgmax2d\n");
	dim3 block(1 , BATCH_SIZE);
	ArgMax2D<<<1, block>>>(output_2);
	cudaDeviceSynchronize();
	
	
	cudaMemcpyFromSymbol( &predicted_labels, arg_max_result, sizeof(int)*BATCH_SIZE);
	cudaDeviceSynchronize();
	
	for(int i = 0; i < BATCH_SIZE;i++)
	{
		
		if( abs(predicted_labels[i] - lab->data[i + BATCH_SIZE*batch_index*lab->width]) < 0.1)
		{
			
			correct_number ++;
		}
		else
		false_number ++;
		
	}
	
	
	/*printf("\nCorrect output : \n");
	DisplayVector2D<<<1, 1>>>(batch_label_data);
	cudaDeviceSynchronize();
	*/
	}
	
	printf("\n\nAccuracy : %f", (float(correct_number)/(correct_number+false_number)*100.0));
	
	printf("\nTamam\n");
	cudaDeviceReset();
program_end = clock();
    printf("\Program execution time : %f\n\n", (double)(program_end- program_start) / CLOCKS_PER_SEC);

	
}
