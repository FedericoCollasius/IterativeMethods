#include "include/metodosIterativos/metodosIterativos.h"
#include <chrono>

using namespace std;
using namespace Eigen;

// Función para generar una matriz aleatoria
MatrixXd generateRandomMatrix(int size) {
    MatrixXd matrix(size, size);
    matrix.setRandom();
    return matrix;
}

// Funcion para generar matriz diagonal dominante con valores chicos para 
// simplificar el cálculo 
MatrixXd generateDominantMatrix(int size) {
    MatrixXd matrix(size, size);
    for(int i=0; i<size; i++){
        int sum = 0;
        for(int j=0; j<size; j++){
            if(i!=j){
                matrix(i,j) = rand() % 500;
                sum += matrix(i,j);
            }
            else{
                matrix(i,j) = 10*size +1;
            }
        }
    }
    return matrix;
}


// Función para generar un vector aleatorio
VectorXd generateRandomVector(int size) {
    VectorXd vector(size);
    int random_int = rand() % 10;
    vector = VectorXd::Constant(size, random_int);
    return vector;
}

// Directamente hacer el producto de la matriz por el vector y 
// comparar con el vector b

// Función para escribir el resultado en un archivo de texto
void writeResultToFile(string filename, string method, double time, double error, VectorXd result) {
    ofstream file(filename, ios::app);
    if (file.is_open()) {
        file << method << "," << time << "," << error << ",";
          for (int i = 0; i < result.size(); i++) {
            file << result(i) << ",";
        }
        file << endl;
        file.close();
    } else {
        cout << "Error al abrir el archivo: " << filename << endl;
    }
}

int main() {
    // Parámetros de prueba
    int matrixSize = 10;
    int numTests = 100;
    int nIter = 50;
    double threshold = 0.00001;
    int checkeoNorma = 100;
    int divThreshold = 10;
    double delta = 10;

    // Generar matrices y vectores aleatorios y realizar las pruebas
    int failed_tests = 0;
    for (int i = 0; i < numTests; i++) {
        MatrixXd A = generateRandomMatrix(matrixSize);
        VectorXd b = generateRandomVector(matrixSize);; 
        //Since Ax = b, x = A^-1 * b
        VectorXd expected = A.inverse() * b;
        VectorXd x0 = VectorXd::Zero(matrixSize);

        auto start = chrono::high_resolution_clock::now();
        VectorXd luResult = resolverLU(A, b);
        auto end = chrono::high_resolution_clock::now();
        double luTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
        double luError = (expected - luResult).norm();

        start = chrono::high_resolution_clock::now();
        VectorXd jSumResult = jSum(A, b, x0, nIter, threshold, checkeoNorma, divThreshold, delta);
        end = chrono::high_resolution_clock::now();
        double jSumTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
        double jSumError = (expected - jSumResult).norm();

        start = chrono::high_resolution_clock::now();
        VectorXd gsSumResult = gsSum(A, b, x0, nIter, threshold, checkeoNorma, divThreshold, delta);
        end = chrono::high_resolution_clock::now();
        double gsSumTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
        double gsSumError = (expected - gsSumResult).norm();

        start = chrono::high_resolution_clock::now();
        VectorXd jMatResult = jMat(A, b, x0, nIter, threshold, checkeoNorma, divThreshold, delta);
        end = chrono::high_resolution_clock::now();
        double jMatTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
        double jMatError = (expected - jMatResult).norm();

        start = chrono::high_resolution_clock::now();
        VectorXd gsMatResult = gsSum(A, b, x0, nIter, threshold, checkeoNorma, divThreshold, delta);
        end = chrono::high_resolution_clock::now();
        double gsMatTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
        double gsMatError = (expected - gsMatResult).norm();

        writeResultToFile("results.csv", "expected", i, matrixSize, expected);
        writeResultToFile("results.csv", "LU", luTime, luError, luResult);
        writeResultToFile("results.csv", "jSum", jSumTime, jSumError, jSumResult);
        writeResultToFile("results.csv", "gsSum", gsSumTime, gsSumError, gsSumResult);
        writeResultToFile("results.csv", "jMat", jMatTime, jMatError, jMatResult);
        writeResultToFile("results.csv", "gsMat", gsMatTime, gsMatError, gsMatResult);
        //write end line to separate tests
        ofstream file("results.csv", ios::app);
        if (file.is_open()) {
            file << endl;
            file.close();
        } else {
            cout << "Error al abrir el archivo: " << "results.csv" << endl;
        }
        failed_tests= jSumResult.hasNaN() + gsSumResult.hasNaN() + jMatResult.hasNaN() + gsMatResult.hasNaN(); 
        
    }
    ofstream file("results.csv", ios::app);
      if (file.is_open()) {
          //Write percentage of failed tests
          //For each run we have 4 methods, so we multiply by 4
          file << "Failed tests: " << failed_tests  << "/" << numTests * 4 << endl;
          file << endl;
          file.close();
      } 

    return 0;
}
