#include "include/metodosIterativos/metodosIterativos.h"
#include <chrono>

using namespace std;
using namespace Eigen;

MatrixXd generateRandomMatrix(int size) {
    MatrixXd matrix(size, size);
    matrix.setRandom();
    return matrix;
}

MatrixXd generateDominantMatrix(int size) {
    MatrixXd matrix(size, size);
    for(int i = 0; i < size; i++){
        int sum = 0;
        for(int j = 0; j < size; j++){
            if(i != j){
                matrix(i, j) = rand() % 500;
                sum += matrix(i, j);
            }
        }
        matrix(i, i) = sum + 1; // Esto garantiza que el elemento de la diagonal sea mayor que la suma de los demás elementos de la fila.
    }
    return matrix;
}

VectorXd generateRandomVector(int size) {
    VectorXd vector(size);
    int random_int = rand() % 10;
    vector = VectorXd::Constant(size, random_int);
    return vector;
}

void writeResultToFile(string filename, int matrixSize, vector<double> tiempos, vector<double> errores) {
    ofstream file(filename, ios::app);
    if (file.is_open()) {
        file << matrixSize << ',' << tiempos[0] << ',' << errores[0] << ',' << tiempos[1] << ',' << errores[1] << ',' << tiempos[2] << ',' << errores[2] << ',' << tiempos[3] << ',' << errores[3] << ',' << tiempos[4] << ',' << errores[4] << endl;
        file.close();
    } else {
        cout << "Error al abrir el archivo: " << filename << endl;
    }
}


int main() {
    // Parámetros de prueba
    int numTests = 100;
    
    int nIter = 50;
    double threshold = 0.00001;
    int checkeoNorma = 10;
    int divThreshold = 10;
    double delta = 10;

    // Generar matrices y vectores aleatorios y realizar las pruebas
    //int failed_tests = 0;
    ofstream file("results.csv", ios::app);
    if (file.is_open()) {
        file << "Tamanio Matrix" << "," << 
        "Promedio Tiempo LU" << "," << "Promedio Error LU" << "," << 
        "Promedio Tiempo JMat" << "," << "Promedio Error JMat" << "," << 
        "Promedio Tiempo GSMat" << "," << "Promedio Error GSMat" << "," <<
        "Promedio Tiempo JSum" << "," << "Promedio Error JSum" << "," <<
        "Promedio Tiempo GSSum" << "," << "Promedio Error GSSum" << endl;
        file.close();
        } else {
        cout << "Error al abrir el archivo: " << "results.csv" << endl;
    }

    vector<int> values = {10, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000};

    for (int i = 0; i < values.size(); i++) {
        int tamanioMatriz = values[i];
        vector<double> tiempos(5, 0.0);
        vector<double> errores(5, 0.0);
        for (int i = 0; i < numTests; i++) {
            MatrixXd A = generateDominantMatrix(tamanioMatriz);
            VectorXd b = generateRandomVector(tamanioMatriz); 
            // Como Ax = b  => x = A^-1 * b
            VectorXd expected = A.inverse() * b;
            VectorXd x0 = generateRandomVector(tamanioMatriz);

            auto start = chrono::high_resolution_clock::now();
            VectorXd luResult = resolverLU(A, b);
            auto end = chrono::high_resolution_clock::now();
            double luTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
            double luError = (expected - luResult).norm();
            tiempos[0] += luTime;
            errores[0] += luError;

            start = chrono::high_resolution_clock::now();
            VectorXd jMatResult = jMat(A, b, x0, nIter, threshold, checkeoNorma, divThreshold, delta);
            end = chrono::high_resolution_clock::now();
            double jMatTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
            double jMatError = (expected - jMatResult).norm();
            tiempos[1] += jMatTime;
            errores[1] += jMatError;

            start = chrono::high_resolution_clock::now();
            VectorXd gsMatResult = gsMat(A, b, x0, nIter, threshold, checkeoNorma, divThreshold, delta);
            end = chrono::high_resolution_clock::now();
            double gsMatTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
            double gsMatError = (expected - gsMatResult).norm();
            tiempos[2] += gsMatTime;
            errores[2] += gsMatError;

            start = chrono::high_resolution_clock::now();
            VectorXd jSumResult = jSum(A, b, x0, nIter, threshold, checkeoNorma, divThreshold, delta);
            end = chrono::high_resolution_clock::now();
            double jSumTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
            double jSumError = (expected - jSumResult).norm();
            tiempos[3] += jSumTime;
            errores[3] += jSumError;

            start = chrono::high_resolution_clock::now();
            VectorXd gsSumResult = gsSum(A, b, x0, nIter, threshold, checkeoNorma, divThreshold, delta);
            end = chrono::high_resolution_clock::now();
            double gsSumTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
            double gsSumError = (expected - gsSumResult).norm();
            tiempos[4] += gsSumTime;
            errores[4] += gsSumError;
        }
        for(int i = 0; i < 5; i++){
            tiempos[i] /= numTests;
            errores[i] /= numTests;
        }
        writeResultToFile("results.csv", tamanioMatriz, tiempos, errores);
    }
    return 0;
}
