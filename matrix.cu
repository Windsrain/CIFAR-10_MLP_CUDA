#include "matrix.cuh"

Matrix::Matrix() {};

Matrix::Matrix(int row, int col) {
    this->height = row;
    this->width = col;
    this->elements = (double *)malloc(row * col * sizeof(double));
}

void Matrix::shuffle(vector<int> ridx) {
    Matrix temp = Matrix(this->height, this->width);
    int i, j, col = this->width;
    for (i = 0; i < this->height; i++)
        for (j = 0; j < this->width; j++)
            temp.elements[i * col + j] = this->elements[ridx[i] * col + j];
    for (i = 0; i < this->height; i++)
        for (j = 0; j < this->width; j++)
        this->elements[i * col + j] = temp.elements[i * col + j];
}

void initialize(Matrix *A, double s) {
    if (s == 0) {
        for (int i = 0; i < A->width * A->height; i++)
            A->elements[i] = 0;
    }
    else {  
        random_device rd;
        default_random_engine gen {rd()};
        normal_distribution<double> dis(0, 1);
        for (int i = 0; i < A->width * A->height; i++) {
            A->elements[i] = dis(gen) / s;
        }
    }
}

void dataCopy(Matrix *A, Matrix B, int s, int t, bool expand) {
    int i, j, width = A->width;
    if (expand == false)
        for (i = 0; i < (t - s); i++)
            for (j = 0; j < width; j++)
                A->elements[i * width + j] = B.elements[((i + s) % B.height) * width + j];
    if (expand == true)
        for (i = 0; i < (t - s); i++)
            for (j = 0; j < 10; j++)
                if (j == B.elements[(i + s) % B.height])
                    A->elements[i * 10 + j] = 1;
                else
                    A->elements[i * 10 + j] = 0;
    
}

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

__device__ double getElement(Matrix *A, int row, int col) {
	return A->elements[row * A->width + col];
}

__device__ void setElement(Matrix *A, int row, int col, double value) {
	A->elements[row * A->width + col] = value;
}

__global__ void matDotKernel(Matrix *A, Matrix *B, Matrix *C, bool trans1, bool trans2) {
	double Cvalue = 0.0;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (trans1 == false && trans2 == false)
        if (row < A->height && col < B->width) {
	        for (int i = 0; i < A->width; ++i)
		        Cvalue += getElement(A, row, i) * getElement(B, i, col);
            setElement(C, row, col, Cvalue);
        }
    if (trans1 == true)
        if (row < A->width && col < B->width) {
            for (int i = 0; i < A->height; i++)
                Cvalue += getElement(A, i, row) * getElement(B, i, col);
            setElement(C, row, col, Cvalue);
        }
    if (trans2 == true)
    if (row < A->height && col < B->height) {
        for (int i = 0; i < A->width; i++)
            Cvalue += getElement(A, row, i) * getElement(B, col, i);
        setElement(C, row, col, Cvalue);
    }      
}

__global__ void matMulKernel(Matrix *A, Matrix *B, Matrix *C) {
    double Cvalue = 0.0;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < A->height && col < A->width) {
        Cvalue = getElement(A, row, col) * getElement(B, row, col);
        setElement(C, row, col, Cvalue);
    }  
}

__global__ void matMulKernel(Matrix *A, double k) {
    double Avalue = 0.0;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < A->height && col < A->width) {
        Avalue = getElement(A, row, col) * k;
        setElement(A, row, col, Avalue);
    }    
}

__global__ void matPlusKernel(Matrix *A, Matrix *B, Matrix *C) {
    double Cvalue = 0.0;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (B->height == 1)
        if (row < A->height && col < A->width) {
            Cvalue = getElement(A, row, col) + getElement(B, 0, col);
            setElement(C, row, col, Cvalue);
        }
}

__global__ void matSubKernel(Matrix *A, Matrix *B, Matrix *C) {
    double Cvalue = 0.0;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < A->height && col < A->width) {
        Cvalue = getElement(A, row, col) - getElement(B, row, col);
        setElement(C, row, col, Cvalue);
    } 
}

__global__ void matSubKernel(double k, Matrix *A, Matrix *B) {
    double Bvalue = 0.0;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < A->height && col < A->width) {
        Bvalue = k - getElement(A, row, col);
        setElement(B, row, col, Bvalue);
    }    
}

__global__ void matReLUKernel(Matrix *A) {
    double Avalue = 0, temp = 0;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < A->height && col < A->width) {
        temp = getElement(A, row, col);
        if (temp > 0)
            Avalue = temp;
        else
            Avalue = 0;
        setElement(A, row, col, Avalue);
    }   
}

__global__ void matDerReLUKernel(Matrix *A) {
    double Avalue = 0, temp = 0;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < A->height && col < A->width) {
        temp = getElement(A, row, col);
        if (temp > 0)
            Avalue = 1;
        else
            Avalue = 0;
        setElement(A, row, col, Avalue);
    }       
}

__global__ void matTanhKernel(Matrix *A) {
    double Avalue = 0, temp = 0;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < A->height && col < A->width) {
        temp = 1.0 / (1 + exp((-2) * getElement(A, row, col)));
        Avalue = 2 * temp - 1;
        setElement(A, row, col, Avalue);
    }
}

__global__ void matExpKernel(Matrix *A) {
    double Avalue = 0;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < A->height && col < A->width) {
        Avalue = exp(getElement(A, row, col));
        setElement(A, row, col, Avalue);
    }
}

__global__ void matPowKernel(Matrix *A, double k) {
    double Avalue = 0;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < A->height && col < A->width) {
        Avalue = pow(getElement(A, row, col), k);
        setElement(A, row, col, Avalue);
    }   
}

__global__ void matSumKernel(Matrix *A, Matrix *B, int axis) {
    double Bvalue = 0;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (axis == 1) {
        if (row < A->height && col < A->width) {
                for (int i = 0; i < A->width; i++)
                    Bvalue += getElement(A, row, i);
                setElement(B, row, 0, Bvalue);
        }        
    }
    if (axis == 0) {
        if (row < A->height && col < A->width) {
            for (int i = 0; i < A->height; i++)
                Bvalue += getElement(A, i, col);
            setElement(B, 0, col, Bvalue);
        }
    }
}

__global__ void matDivKernel(Matrix *A, Matrix *B) {
    double Avalue = 0;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < A->height && col < A->width) {
        Avalue = getElement(A, row, col) / getElement(B, row, 0);
        setElement(A, row, col, Avalue);
    }  
}

__global__ void matcountEqual1(Matrix *A, Matrix *B, int *cnt) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < A->height && col < A->width)
        if ((getElement(A, row, col) == getElement(B, row, col)) && (getElement(A, row, col) == 1))
            *cnt = *cnt + 1;
}
