#include "cifar10_reader.cuh"

void readImages(string fileName, Matrix &labels, Matrix &images, int len, int pos) {
    ifstream file(fileName, ios::binary);
    if (!file.is_open())
        cout << "Error Opening File!\n" << endl;
    for (int i = 0; i < len; i++) {
        unsigned char pixel = 0;
        file.read((char *)&pixel, sizeof(pixel));
        labels.elements[i + pos] = (double)pixel;
        for (int j = 0; j < 3072; j++) {
            file.read((char *)&pixel, sizeof(pixel));
            images.elements[(i + pos) * 3072 + j] = (double)pixel / 255;
        }
    }
    file.close();
}

unordered_map<string, Matrix> readData() {
    extern int trainLen;
    extern int testLen;

    Matrix trainImages = Matrix(trainLen, 32 * 32 * 3);
    Matrix trainLabels = Matrix(trainLen, 1);
    Matrix testImages = Matrix(testLen, 32 * 32 * 3);
    Matrix testLabels = Matrix(testLen, 1);

    readImages("./data/data_batch_1.bin", trainLabels, trainImages, 10000, 0);
    readImages("./data/data_batch_2.bin", trainLabels, trainImages, 10000, 10000);
    readImages("./data/data_batch_3.bin", trainLabels, trainImages, 10000, 20000);
    readImages("./data/data_batch_4.bin", trainLabels, trainImages, 10000, 30000);
    readImages("./data/data_batch_5.bin", trainLabels, trainImages, 10000, 40000);
    readImages("./data/test_batch.bin", testLabels, testImages, testLen, 0);

    unordered_map<string, Matrix> dataMap;

    dataMap.insert({"trainImages", trainImages});
    dataMap.insert({"trainLabels", trainLabels});
    dataMap.insert({"testImages", testImages});
    dataMap.insert({"testLabels", testLabels});

    return dataMap;
}