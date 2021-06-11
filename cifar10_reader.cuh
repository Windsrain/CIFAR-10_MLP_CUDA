#ifndef CIFAR10_READER_CUH
#define CIFAR10_READER_CUH
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include "matrix.cuh"

using namespace std;
void readImages(string fileName, Matrix &labels, Matrix &images, int len, int pos);
unordered_map<string, Matrix> readData();

#endif