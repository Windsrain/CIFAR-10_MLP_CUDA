#include <vector>
#include <algorithm>
#include "cifar10_reader.cuh"
#include "matrix.cuh"

int trainLen = 50000;
int testLen = 10000;
int cnt, loss;
const int batch = 200;
double epsilon = 0.0001;
double regLambda = 0.00;
int numExamples, nbsPerEpoch, inputDim, nnHdim[5] = {0, 2048, 1024, 512, 10};

Matrix *Xb, *Yb, *Xt, *Yt, *Y_pred, *W[4], *B[4], *a[4], *delta[4], *dW[4], *dB[4];
Matrix *softMaxSum;
Matrix X_train, Y_train, X_test, Y_test;

void generate();
void predict();
void calLoss();
void forwardPropagation();
void backPropagation();
void trainModel(int numPasses, bool printLoss);

int main() {
    cout.precision(16);
    unordered_map<string, Matrix> dataMap = readData();
    X_train = dataMap["trainImages"];
    Y_train = dataMap["trainLabels"];
    X_test = dataMap["testImages"];
    Y_test = dataMap["testLabels"];
    printf("(%d, %d) (%d, %d)\n", X_train.height, X_train.width, Y_train.height, Y_train.width);
    printf("(%d, %d) (%d, %d)\n", X_test.height, X_test.width, Y_test.height, Y_test.width);
    numExamples = X_train.height;
    inputDim = X_train.width;
    nbsPerEpoch = (int)(numExamples / batch);
    generate();
    trainModel(10000, true);

    return 0;
}

void generate() {
    printf("input dim: %d\n", inputDim);
    // 初始化各矩阵，为其分配共享内存
    nnHdim[0] = inputDim;
    cudaMallocManaged((void **)&(Xb), sizeof(Matrix));
    Xb->height = batch; Xb->width = inputDim;
    cudaMallocManaged((void **)&(Yb), sizeof(Matrix));
    Yb->height = batch; Yb->width = 10;
    cudaMallocManaged((void **)&(Xt), sizeof(Matrix));
    Xt->height = batch; Xt->width = inputDim;
    cudaMallocManaged((void **)&(Yt), sizeof(Matrix));
    Yt->height = batch; Yt->width = 10;
    cudaMallocManaged((void **)&(Y_pred), sizeof(Matrix));
    Y_pred->height = batch; Y_pred->width = 10;
    cudaMallocManaged((void **)&(Xb->elements), batch * inputDim * sizeof(double));
    cudaMallocManaged((void **)&(Yb->elements), batch * 10 * sizeof(double));
    cudaMallocManaged((void **)&(Xt->elements), batch * inputDim * sizeof(double));
    cudaMallocManaged((void **)&(Yt->elements), batch * 10 * sizeof(double));
    cudaMallocManaged((void **)&(Y_pred->elements), batch * 10 * sizeof(double));
    for (int i = 0; i < 4; i++) {
        int row = nnHdim[i], col = nnHdim[i + 1];
        double std = sqrt(col);
        cudaMallocManaged((void **)&(W[i]), sizeof(Matrix));
        W[i]->width = col; W[i]->height = row;
        cudaMallocManaged((void **)&(B[i]), sizeof(Matrix));
        B[i]->width = col; B[i]->height = 1;
        cudaMallocManaged((void **)&(a[i]), sizeof(Matrix));
        a[i]->width = col; a[i]->height = batch;
        cudaMallocManaged((void **)&(delta[i]), sizeof(Matrix));
        delta[i]->width = col; delta[i]->height = batch;
        cudaMallocManaged((void **)&(dW[i]), sizeof(Matrix));
        dW[i]->width = col; dW[i]->height = row;
        cudaMallocManaged((void **)&(dB[i]), sizeof(Matrix));
        dB[i]->width = col; dB[i]->height = 1;
        cudaMallocManaged((void **)&(W[i]->elements), row * col * sizeof(double));
        cudaMallocManaged((void **)&(B[i]->elements), 1 * col * sizeof(double));
        cudaMallocManaged((void **)&(a[i]->elements), batch * col * sizeof(double));
        cudaMallocManaged((void **)&(delta[i]->elements), batch * col * sizeof(double));
        cudaMallocManaged((void **)&(dW[i]->elements), row * col * sizeof(double));
        cudaMallocManaged((void **)&(dB[i]->elements), row * col * sizeof(double));
        printf("fc: %d -> %d\n", row, col);
        initialize(W[i], std);
        initialize(B[i], 0);
    }
    cudaMallocManaged((void **)&(softMaxSum), sizeof(Matrix));
    softMaxSum->width = 1; softMaxSum->height = batch;
    cudaMallocManaged((void **)&(softMaxSum->elements), batch * 1 * sizeof(double));
    cudaDeviceSynchronize();
}

void predict() {
    dim3 blockSize(32, 32);
	dim3 gridSize(32, 32);   
    //输入层：Xb，隐藏层：a[0]，a[1]，a[2]，输出层：a[3]   
    matDotKernel <<<gridSize, blockSize>>> (Xt, W[0], a[0]);
    matPlusKernel <<<gridSize, blockSize>>> (a[0], B[0], a[0]);
    matReLUKernel <<<gridSize, blockSize>>> (a[0]);
    matDotKernel <<<gridSize, blockSize>>> (a[0], W[1], a[1]);
    matPlusKernel <<<gridSize, blockSize>>> (a[1], B[1], a[1]);
    matReLUKernel <<<gridSize, blockSize>>> (a[1]);
    matDotKernel <<<gridSize, blockSize>>> (a[1], W[2], a[2]);
    matPlusKernel <<<gridSize, blockSize>>> (a[2], B[2], a[2]);
    matReLUKernel <<<gridSize, blockSize>>> (a[2]);
    matDotKernel <<<gridSize, blockSize>>> (a[2], W[3], a[3]);
    matPlusKernel <<<gridSize, blockSize>>> (a[3], B[3], a[3]);
    cudaDeviceSynchronize();
    for (int i = 0; i < a[3]->height; i++) {
        int maxIndex = 0; 
        double maxValue = a[3]->elements[i * a[3]->width];
        for (int j = 0; j < a[3]->width; j++)
            if (a[3]->elements[i * a[3]->width + j] > maxValue) {
                maxIndex = j;
                maxValue = a[3]->elements[i * a[3]->width + j];
            }
        if (Yt->elements[i * a[3]->width + maxIndex])
            cnt++;
    }
    cudaDeviceSynchronize();
}

void calLoss() {
    dim3 blockSize(32, 32);
	dim3 gridSize(32, 32);   
    //输入层：Xb，隐藏层：a[0]，a[1]，a[2]，输出层：a[3]   
    matDotKernel <<<gridSize, blockSize>>> (Xb, W[0], a[0]);
    matPlusKernel <<<gridSize, blockSize>>> (a[0], B[0], a[0]);
    matReLUKernel <<<gridSize, blockSize>>> (a[0]);
    matDotKernel <<<gridSize, blockSize>>> (a[0], W[1], a[1]);
    matPlusKernel <<<gridSize, blockSize>>> (a[1], B[1], a[1]);
    matReLUKernel <<<gridSize, blockSize>>> (a[1]);
    matDotKernel <<<gridSize, blockSize>>> (a[1], W[2], a[2]);
    matPlusKernel <<<gridSize, blockSize>>> (a[2], B[2], a[2]);
    matReLUKernel <<<gridSize, blockSize>>> (a[2]);
    matDotKernel <<<gridSize, blockSize>>> (a[2], W[3], a[3]);
    matPlusKernel <<<gridSize, blockSize>>> (a[3], B[3], a[3]);
    cudaDeviceSynchronize();
    for (int i = 0; i < a[3]->height; i++) {
        int maxIndex = 0; 
        double maxValue = a[3]->elements[i * a[3]->width];
        for (int j = 0; j < a[3]->width; j++)
            if (a[3]->elements[i * a[3]->width + j] > maxValue) {
                maxIndex = j;
                maxValue = a[3]->elements[i * a[3]->width + j];
            }
        if (Yb->elements[i * a[3]->width + maxIndex])
            loss++;
    }
    cudaDeviceSynchronize();
}

void forwardPropagation() {
    dim3 blockSize(32, 32);
	dim3 gridSize(32, 32);    
    //输入层：Xb，隐藏层：a[0]，a[1]，a[2]，输出层：a[3]    
    matDotKernel <<<gridSize, blockSize>>> (Xb, W[0], a[0]);
    matPlusKernel <<<gridSize, blockSize>>> (a[0], B[0], a[0]);
    matReLUKernel <<<gridSize, blockSize>>> (a[0]);
    matDotKernel <<<gridSize, blockSize>>> (a[0], W[1], a[1]);
    matPlusKernel <<<gridSize, blockSize>>> (a[1], B[1], a[1]);
    matReLUKernel <<<gridSize, blockSize>>> (a[1]);
    matDotKernel <<<gridSize, blockSize>>> (a[1], W[2], a[2]);
    matPlusKernel <<<gridSize, blockSize>>> (a[2], B[2], a[2]);
    matReLUKernel <<<gridSize, blockSize>>> (a[2]);
    matDotKernel <<<gridSize, blockSize>>> (a[2], W[3], a[3]);
    matPlusKernel <<<gridSize, blockSize>>> (a[3], B[3], a[3]);
    //输出层使用softmax
    matExpKernel <<<gridSize, blockSize>>> (a[3]);
    matSumKernel <<<gridSize, blockSize>>> (a[3], softMaxSum, 1);
    matDivKernel <<<gridSize, blockSize>>> (a[3], softMaxSum);
    cudaDeviceSynchronize();
}

void backPropagation() {
    dim3 blockSize(32, 32);
    dim3 gridSize(32, 32);
    //反向传播
    matSubKernel <<<gridSize, blockSize>>> (a[3], Yb, delta[3]);
    cudaDeviceSynchronize();

    matDotKernel <<<gridSize, blockSize>>> (a[2], delta[3], dW[3], true);   
    matSumKernel <<<gridSize, blockSize>>> (delta[3], dB[3], 0);
    cudaDeviceSynchronize();

    matDerReLUKernel <<<gridSize, blockSize>>> (a[2]);
    matDotKernel <<<gridSize, blockSize>>> (delta[3], W[3], delta[2], false, true);      
    matMulKernel <<<gridSize, blockSize>>> (delta[2], a[2], delta[2]);
    matDotKernel <<<gridSize, blockSize>>> (a[1], delta[2], dW[2], true);
    matSumKernel <<<gridSize, blockSize>>> (delta[2], dB[2], 0);
    cudaDeviceSynchronize();

    matDerReLUKernel <<<gridSize, blockSize>>> (a[1]);
    matDotKernel <<<gridSize, blockSize>>> (delta[2], W[2], delta[1], false, true);      
    matMulKernel <<<gridSize, blockSize>>> (delta[1], a[1], delta[1]);
    matDotKernel <<<gridSize, blockSize>>> (a[0], delta[1], dW[1], true);
    matSumKernel <<<gridSize, blockSize>>> (delta[1], dB[1], 0);
    cudaDeviceSynchronize();

    matDerReLUKernel <<<gridSize, blockSize>>> (a[0]);
    matDotKernel <<<gridSize, blockSize>>> (delta[1], W[1], delta[0], false, true);      
    matMulKernel <<<gridSize, blockSize>>> (delta[0], a[0], delta[0]);
    matDotKernel <<<gridSize, blockSize>>> (Xb, delta[0], dW[0], true);
    matSumKernel <<<gridSize, blockSize>>> (delta[0], dB[0], 0);
    cudaDeviceSynchronize();

    //梯度更新
    for (int i = 0; i < 4; i++) {
        matMulKernel <<<gridSize, blockSize>>> (dW[i], epsilon);
        matMulKernel <<<gridSize, blockSize>>> (dB[i], epsilon);
        matSubKernel <<<gridSize, blockSize>>> (W[i], dW[i], W[i]);
        matSubKernel <<<gridSize, blockSize>>> (B[i], dB[i], B[i]);
        cudaDeviceSynchronize();
    }
}

void trainModel(int numPasses, bool printLoss) {
    int i;
    for (i = 0; i <= numPasses; i++) {
        int j = i % nbsPerEpoch;
        //每训练一次完整的训练集，就重新打乱训练集
        if (j == 0) {
            vector<int> ridx(numExamples);
            int k;
            for (k = 0; k < numExamples; k++)
                ridx[k] = k;
            random_shuffle(ridx.begin(), ridx.end());
            X_train.shuffle(ridx);
            Y_train.shuffle(ridx);
        }
        //获取训练集的一个batch
        dataCopy(Xb, X_train, j * batch, (j + 1) * batch);
        dataCopy(Yb, Y_train, j * batch, (j + 1) * batch, true);
        forwardPropagation();
        backPropagation();
        if (printLoss && (i % 100 == 0)) {
            epsilon *= 0.99;
            cnt = 0;
             //将测试集分成batch依次预测
            for (int k = 0; k < (int)(X_test.height / batch); k++) {
                dataCopy(Xt, X_test, k * batch, (k + 1) * batch);
                dataCopy(Yt, Y_test, k * batch, (k + 1) * batch, true);   
                predict();     
                cudaDeviceSynchronize();
            }
            double accuracy = (cnt * 1.0 / X_test.height);
            struct tm *p;
            time_t t = time(0);
            p = localtime(&t);
            printf("%02d:%02d:%02d testing accuracy after iteration %d: %.2lf%%\n", p->tm_hour, p->tm_min, p->tm_sec, i, accuracy * 100);
        }

        //经过一个完成的测试集，输出train loss
        if (printLoss && (j == 0) && (i != 0)) {
            double accuracy = (loss * 1.0 / X_train.height);
            struct tm *p;
            time_t t = time(0);
            p = localtime(&t);
            printf("\n%02d:%02d:%02d train loss after iteration %d: %.2lf%%\n\n", p->tm_hour, p->tm_min, p->tm_sec, i, accuracy * 100);
            loss = 0;
        }
        
        calLoss();
    }
}