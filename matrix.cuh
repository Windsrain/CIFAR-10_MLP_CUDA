#ifndef MATRIX_CUH
#define MATRIX_CUH
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <random>
#include <ctime>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

struct Matrix {
    int width, height;
    double *elements;
    Matrix();
    Matrix(int row, int col);
    void shuffle(vector<int> ridx);
};

void initialize(Matrix *A, double s);
void dataCopy(Matrix *A, Matrix B, int s, int t, bool expand = false);
double sigmoid(double x);
__device__ double getElement(Matrix *A, int row, int col);
__device__ void setElement(Matrix *A, int row, int col, double value);
__global__ void matDotKernel(Matrix *A, Matrix *B, Matrix *C, bool trans1 = false, bool trans2 = false);
__global__ void matMulKernel(Matrix *A, Matrix *B, Matrix *C);
__global__ void matMulKernel(Matrix *A, double k);
__global__ void matPlusKernel(Matrix *A, Matrix *B, Matrix *C);
__global__ void matSubKernel(Matrix *A, Matrix *B, Matrix *C);
__global__ void matSubKernel(double k, Matrix *A, Matrix *B);
__global__ void matReLUKernel(Matrix *A);
__global__ void matDerReLUKernel(Matrix *A);
__global__ void matTanhKernel(Matrix *A);
__global__ void matExpKernel(Matrix *A);
__global__ void matPowKernel(Matrix *A, double k);
__global__ void matSumKernel(Matrix *A, Matrix *B, int axis);
__global__ void matDivKernel(Matrix *A, Matrix *sum);
__global__ void matcountEqual1(Matrix *A, Matrix *B, int *cnt);

#endif