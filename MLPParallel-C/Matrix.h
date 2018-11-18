#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED
/*
    This file contains the required functions related Matrix operations used in Perceptron algorithm.



    Implemented Functions:
        Vector2D * CopyVector(Vector2D * vec)
        Vector2D * VectorSlice2D(Vector2D * vec, int row_start, int row_end, int column_start, int column_end)
        Vector2D * ScalarMinusVector2D(float value, Vector2D * vec)
        Vector2D * TransposeVector2D(Vector2D * vec)
        Vector2D * MatrixSubtract(Vector2D * v1, Vector2D * v2)
        Vector2D * MatrixAdd(Vector2D * v1, Vector2D * v2)
        Vector2D * MatrixPairwiseProduct(Vector2D * f1, Vector2D * f2)
        Vector2D * ArgMax2D(Vector2D * vector)
        double Sum2D(Vector2D * vec)
        Vector2D * Log2D(Vector2D * vec)
        Vector2D * ColumnVectorAdd(Vector2D * vec, float value)
        Vector2D * MatrixProductWithRowAdd(Vector2D * f1, Vector2D * f2, float value)
        Vector2D * MatrixProduct(Vector2D * f1, Vector2D * f2 )
        Vector2D * ScalarMatrixProduct(float scalar, Vector2D * vector)
        Vector2D * CreateVector2D(float * data, int height, int width)
        void DestroyVector2D(Vector2D * vec)
        void DisplayVector2D(Vector2D * vector)
        void VectorInfo(Vector2D * vec)



    All vectors have been represented as a two dimensional vectors even the one dimensional ones like matrices.

    X = [x0 x1 x2 x3 ... ] is represented in (1, len(X)) dimension where the row number is one and the column
    number is the X's vector length.

    X = [ x0 ] is represented as (4, 1) dimensional vector where row number is 4 and the column number is 1
        [ x1 ]
        [ x3 ]
        [ x4 ]


*/

// C language does not contain a boolean type, we are defining our type...
#define FALSE 0
#define TRUE 1
typedef char bool;

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


// This function copies an existing vector/matrix and returns its address...
Vector2D * CopyVector(Vector2D * vec)
{
    // We are allocating space for the new copy vector/matrix...
    float * d = malloc(vec->height*vec->width*sizeof(float));
    // We are loading copied vector's/matrix's data to our new vector/matrix...
    for(int a=0; a<vec->height*vec->width; a++)
        d[a] = vec->data[a];

    // A new structure area for the new vector/matrix is allocated...
    // We are using malloc because we would like variable to live after termination of the function...
    // So this variables should not be defined as a local variable...
    Vector2D * temp = malloc(sizeof(Vector2D));

    // We are assigning original vector's/matrix's structure
    temp->height = vec->height;
    temp->width  = vec->width;
    temp->size   = vec->size;
    // We are assinging the newly created vector/matrix data area...
    temp->data = d;
    return temp;
}


// This function slices any vector/matrix in two dimension...
// This functions is inspired by the python's array slice mechanism...
/*
    X = [ 1 2 3 4 5 ]     height = 4, width = 5
        [ 6 7 8 9 0 ]
        [ 0 1 2 3 4 ]
        [ 5 6 7 8 9 ]

    VectorSlice2D( X, 1, 3, 2, 5) -> [ 8 9 0 ]   X is sliced in row from first row to 3th (3th row is excluded)
                                     [ 2 3 4 ]   in column from second to fifth (fifth column is excluded)


*/
// Although name is specific for vector, it can also be used to slice matrices...
Vector2D * VectorSlice2D(Vector2D * vec, int row_start, int row_end, int column_start, int column_end)
{
    // A new vector/matrix is created for the slice ...
    Vector2D * temp = malloc(sizeof(Vector2D));
    temp->height = row_end - row_start ;
    temp->width = column_end - column_start;
    temp->data = malloc(sizeof(float)*temp->height*temp->width);
    temp->size = temp->height;

    // Sliced part of the vector/matrix is loaded to newly created vector/matrix...
    // h is row index.... Column index starts from row_start to row_end
    for(int h = row_start ; h < row_end; h++)
    {
        // The row and column
        // w is column index... column index starts from column_start to column_end
        for(int w = column_start; w < column_end; w++)
        {
            // Because the index could be initiated from nonzero, we have to shift the index to store the data
            // properly in slice vector/matrix
            temp->data[((h-row_start)*temp->width)+w-column_start] = vec->data[h*vec->width+w];
        }
    }

    return temp;
}



/*
    This function creates a vector/matrix via new(i,j) = scalar - vec(i,j) where i represents ith row
                                                                           j represents jth column
    X = [ 1 2 3 ]      ScalarMinusVector2D(5, X) -> [ 5-1 5-2 5-3 ] -> [ 4 3 2 ]
        [ 4 5 6 ]                                   [ 5-4 5-5 5-6 ]    [ 1 0 -1]

*/
Vector2D * ScalarMinusVector2D(float value, Vector2D * vec)
{
    // A new vector/matrix is created for the result...
    Vector2D * temp = malloc(sizeof(Vector2D));
    // Dimensions of the new vector/matrix is the same as inputed vector's/matrix's dimensions
    temp->height = vec->height;
    temp->width = vec->width;
    temp->data = malloc(sizeof(float)*temp->height*temp->width);
    temp->size = temp->height;

    // New created vector/matrix is filled by the operation declared above...
    for(int h = 0; h< vec->height; h++)
    {
        for(int w=0; w<vec->width; w++)
        {
            temp->data[h*vec->width+w] = 1 - temp->data[h*vec->width+w];
        }
    }
    return temp;
}


// Function performs vector/matrix transpose operation...
Vector2D * TransposeVector2D(Vector2D * vec)
{
    // A new vector/matrix is created for the transpose
    float * d = malloc(vec->height*vec->width*sizeof(float));
    Vector2D * temp = malloc(sizeof(Vector2D));
    temp->height = vec->width;
    temp->width = vec->height;
    temp->size = temp->height;
    temp->data = d;

    // we are loading data to transpose matrix...
    for(int h=0; h<vec->height;  h++)
    {
            for(int w=0; w<vec->width; w++)
            {
                // w and h indices are switched...
                temp->data[w*vec->height+h] = vec->data[h*vec->width+w];
            }

    }


    return temp;
}


// This function is used to subtract a vectors/matrices from other
// This function modifies the first vector/matrix such that the subtraction is stored in the first parameter
// No new vector/matrix is created... No new vector/Matrix is not created to reduce memory operations...
Vector2D * MatrixSubtract(Vector2D * v1, Vector2D * v2)
{
    // For subtraction, dimensions of the both vector/matrix must be same...
    if((v1->height != v2->height) ||(v1->width != v2->width))
    {
        printf("\n\nMatrixSubtract dimensions are not equal...\n\n");
        exit(0);
    }

    // We are overriding the first vector/matrix with the result...
    for(int i=0; i<v1->height*v1->width; i++)
    {
        v1->data[i] = v1->data[i] - v2->data[i];
    }
    return v1;
}

// This function adds two vectors/matrices pairwisely...
// First input vector/matrix is overrided with the result...
// No new vector/Matrix is not created to reduce memory operations...
Vector2D * MatrixAdd(Vector2D * v1, Vector2D * v2)
{
    // Vector/Matrix dimensions of both input must be the same to perform addition...
     if((v1->height != v2->height) || (v1->width != v2->width))
    {
        printf("\n\nMatrixAdd dimensions are not equal...\n\n");
        exit(0);
    }

    // We are overriding the first vector/matrix with the result...
    for(int i=0; i<v1->height*v1->width; i++)
    {
        v1->data[i] = v1->data[i] + v2->data[i];
    }
    return v1;
}


// This function multiplies two vectors/matrices elementwise...
// The first vector/matrix is overrided with the result...
// No new vector/Matrix is not created to reduce memory operations...
/*

X = [ 1 2 3 ]   Y = [ 6 5 4 ]   MatrixPairwiseProduct(X,Y)-> [1*6 2*5 3*4]  -> [6 10 12]
    [ 4 5 6 ]       [ 3 2 1 ]                                [4*3 5*2 6*1]     [12 10 6]


*/
Vector2D * MatrixPairwiseProduct(Vector2D * f1, Vector2D * f2)
{
    // Vectors/Matrices dimensions must be the same...
     if((f1->height != f2->height) ||(f1->width != f2->width))
    {
        printf("\n\nMatrixPairwiseProduct dimensions are not equal...\n\n");
        exit(0);
    }

    // First vector/matrix is overrided by the result...
    for(int h=0; h < f1->height; h++)
    {
        for(int w=0; w<f1->width; w++)
        {
            f1->data[h*f1->width+w] = f1->data[h*f1->width+w]*f2->data[h*f1->width+w];
        }

    }
    return f1;
}

// This functions is used to find the maximum valued index in one row... - In Python numpy.argmax()
// Max values indices are found for each row of the matrix/vector
// The result is stored in a newly created vector...
/*
    X = [ 0 1 5 3 ] ArgMax2D(X) -> [ 2 ]
        [ 2 7 6 1 ]                [ 1 ]
        [ 1 2 3 8 ]                [ 3 ]


*/
Vector2D * ArgMax2D(Vector2D * vector)
{
    // The result will be stored in a column vector...
    Vector2D * temp = malloc(sizeof(Vector2D));
    temp->height = vector->height;
    temp->width = 1;
    temp->size = temp->height;
    temp->data = malloc(temp->height*temp->width*sizeof(float));


    float max_number = 0;
    int max_index = 0;
    for(int h=0; h < vector->height; h++)
    {
        // For each row, the maximum valued index is found
        max_number = vector->data[h*vector->width];
        max_index = 0;
        for(int w=0; w < vector->width; w++)
        {
            if(vector->data[h*vector->width+w]>max_number){
              max_number = vector->data[h*vector->width+w];
              max_index = w;
            }
        }

        // The index is written to result vector...
        temp->data[h] = max_index;

    }
    return temp;
}


// This function sums all elements of a vector/matrix...
/*
    X = [ 1 2 3 ]   Sum2D(X) -> 1+2+3+7+9+8 = 30
        [ 7 9 8 ]


*/
double Sum2D(Vector2D * vec)
{
    double sum = 0.0;
    for(int i=0; i < vec->width*vec->height; i++)
    {
        sum += vec->data[i];
    }
    return sum;
}

// This function performs log() operaion for each element of vector/matrix
// The original matrix is overrided with the result
/*
    X = [ 1 2 3 ]    Log2D(X) -> [ log(1) log(2) log(3) ]
        [ 4 5 6 ]                [ log(4) log(5) log(6) ]

*/

Vector2D * Log2D(Vector2D * vec)
{
    for(int i = 0; i< vec->width*vec->height; i++)
        {

            vec->data[i] = log(vec->data[i]);

        }
    return vec;
}


// This function adds a column vector to a given matrix. 2 the result a new vector/matrix is created...
// New column vector is put to 0th column vector place ...
// The elements of the column vector is specified by the value parameter. All elements of the column vector is same.
/*
    X = [ 2 3 4 ]   ColumnVectorAdd(X , 4) ->  [ 4 2 3 4 ]
        [ 7 8 9 ]                              [ 4 7 8 9 ]
*/
Vector2D * ColumnVectorAdd(Vector2D * vec, float value)
{
    // Result matrix/vector is created...
    Vector2D * new_result = malloc(sizeof(Vector2D));
    new_result->data = malloc(vec->height*(vec->width+1)*sizeof(float));
    new_result->height = vec->height;
    new_result->width = vec->width+1;
    new_result->size = new_result->height;
    for(int h = 0; h<new_result->height; h++)
    {
        // 0th index of the new result matrix/vector is the passed value...
        new_result->data[h*new_result->width] = value;
        for(int w = 1; w<new_result->width; w++)
        {
            // To add the original matrix, the required index shift should be performed...
            new_result->data[h*new_result->width+w] = vec->data[h*vec->width + w-1];
        }
    }

    return new_result;
}


// This functions performs a matrix multiplication between two vector s/matrices and adds a column vector to 0th column...
// This function implements two separate function operation in one attempt.
// Same result could be obtained by ColumnVectorAdd(MatrixProduct(X, Y), value)...
// For the result a new matrix/vector is created....
/*
    X = [ 1 2 ]  Y = [ 3 4 5 ]  MatrixProductWithRowAdd(X, Y, 2.0) -> [(1*3+2*1) (1*4+2*8) (1*5+2*9)]
        [ 4 5 ]      [ 1 8 9 ]                                        [(4*3+5*1) (4*4+5*8) (4*5+5*9)]

    -> [ 5 20 23 ]    AddColumn -> [ 2 5 20 23 ]
       [ 17 56 65 ]                [ 2 17 56 65 ]

*/
Vector2D * MatrixProductWithRowAdd(Vector2D * f1, Vector2D * f2, float value)
{
    // For matrix multiplication, the column number of the first vector/matrix must be same with row number of the second vector/matrix...
    if(f1->width != f2->height)
    {
        printf("MatrixProductWithRowAdd dimension is not equal...");
        exit(0);
    }

    // New result matrix/vector is created...
    Vector2D * new_result = malloc(sizeof(Vector2D));
    new_result->data = malloc(f1->height*(f2->width+1)*sizeof(float));
    new_result->height = f1->height;
    // 1 must be added because 0th column will be filled with value...
    new_result->width = f2->width+1;
    new_result->size = new_result->height;

    // Matrix multiplication and adding a column vector to result is performed..
    float sum = 0;
    for(int row1 = 0;row1 < f1->height; row1++ )
    {
        for(int column2=0; column2 < f2->width; column2++)
        {

            // If column number is 0, we are putting the value...
            if(column2 == 0)
             new_result->data[row1*(f2->width+1) ] = value;

            sum = 0;
            // one row vector of the first matrix/vector and a column vector of the second vector/matrix is
            // multiplied with dot product and the result is stored...
            for(int index = 0; index < f1->width;index++)
            {
                sum += f1->data[row1*f1->width+index]*
                f2->data[index*f2->width+column2];
            }

             new_result->data[row1*(f2->width+1) + column2+1] = sum;
        }
    }

    return new_result;

}

// This function performs matrix multiplication and the result is stored in a new vector/matrix
/*
    X = [ 1 2 ]    y = [ 4 5 6 ]  MatrixProduct(X, Y) -> [(1*4+2*3) (1*5+2*2) (1*6+2*1)] -> [10 9 8]
        [ 5 8 ]        [ 3 2 1 ]                         [(5*4+8*3) (5*5+8*2) (5*6+8*1)]    [44 41 38]


*/
Vector2D * MatrixProduct(Vector2D * f1, Vector2D * f2 )
{
    // Row dimension of the first vector/matrix must be the same with the column dimension of the second matrix/vector
     if(f1->width != f2->height)
    {
        printf("MatrixProduct dimension is not equal...");
        exit(0);
    }

    // A new vector/matrix is allocated for the result...
    Vector2D * new_result = malloc(sizeof(Vector2D));
    new_result->data = malloc(f1->height*f2->width*sizeof(float));
    new_result->height = f1->height;
    new_result->width = f2->width;
    new_result->size = new_result->height;


    float sum = 0;
    for(int row1 = 0;row1 < f1->height; row1++ )
    {
        for(int column2=0; column2 < f2->width; column2++)
        {

            // One row column from first vector/matrix is dot producted via one column vector of the second matrix/vector
            sum = 0;
            for(int index = 0; index < f1->width;index++)
            {
                sum += f1->data[row1*f1->width+index]*
                f2->data[index*f2->width+column2];
            }
            // The result of dot product is stored in result matrix/vector...
             new_result->data[row1*f2->width + column2] = sum;

        }

    }

    return new_result;

}



// This function multiplies every element of a vector/matrix with a scalar, the result is stored passed matrix/vector
// A new matrix/vector is not created for the result...
/*
    X = [ 1 4 3 ]   ScalarMatrixProduct(2, X) -> [2*1 2*4 2*3] -> [2 8 6]
        [ 2 3 1 ]                                [2*2 2*3 2*1]    [4 6 2]


*/
Vector2D * ScalarMatrixProduct(float scalar, Vector2D * vector)
{
    for(int i=0; i<vector->height*vector->width;i++)
    {
        //Passed matrix/vector is overrided with the result...
        vector->data[i] *= scalar;
    }
return vector;
}



// This function creates the Vector2d structure to store matrix/vector...
// The matrix/vector data is passed as a parameter...
// The row and column dimensions of the vector/matrix is provided...
Vector2D * CreateVector2D(float * data, int height, int width)
{
    // A new structure is allocated in memory for matrix/vector...
    Vector2D * temp = (Vector2D *)malloc(sizeof(struct Vector2D));
    temp->data = data;
    temp->height = height;
    temp->width = width;
    return temp;
};


// This function deallocates a vector/matrix...
void DestroyVector2D(Vector2D * vec)
{
    // First the matrix/vector data is deallocated...
    free(vec->data);
    // Later the memory allocated for the matrix/vector structure is deleted...
    free(vec);

}

// It displays the content of vector/matrix
void DisplayVector2D(Vector2D * vector)
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

// This function displays the dimension of the given vector/matrix
void VectorInfo(Vector2D * vec)
{
    printf("\nvector height : %d - width : %d\n", vec->height, vec->width);

}
#endif // MATRIX_H_INCLUDED
