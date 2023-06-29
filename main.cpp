#include "include/metodosIterativos/metodosIterativos.h"
#include <fstream>
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
    // for each cell except diagonal, add a random number between 0 and 10
    for(int i=0; i<size; i++){
        for(int j=0; j<size; j++){
            if(i!=j){
                matrix(i,j) = rand() % 10;
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

// Función para escribir el resultado en un archivo de texto
void writeResultToFile(string filename, string method, double time, double error) {
    ofstream file(filename, ios::app);
    if (file.is_open()) {
        file << method << "," << time << "," << error << endl;
        file.close();
    } else {
        cout << "Error al abrir el archivo: " << filename << endl;
    }
}

int main() {
    // Parámetros de prueba
    int matrixSize = 100;
    int numTests = 10;
    int nIter = 500;
    double threshold = 0.00000000001;
    int checkeoNorma = 100;

    // Generar matrices y vectores aleatorios y realizar las pruebas
    for (int i = 0; i < numTests; i++) {
        MatrixXd A = generateRandomMatrix(matrixSize);
        VectorXd b = generateRandomVector(matrixSize);
        VectorXd x0 = VectorXd::Zero(matrixSize);

        auto start = chrono::high_resolution_clock::now();
        VectorXd expected = A.lu().solve(b);
        auto end = chrono::high_resolution_clock::now();
        double luTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;

        start = chrono::high_resolution_clock::now();
        VectorXd jsumResult = jSum(A, b, x0, nIter, threshold, checkeoNorma);
        end = chrono::high_resolution_clock::now();
        double jsumTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
        double jsumError = (expected - jsumResult).norm();

        start = chrono::high_resolution_clock::now();
        VectorXd gsSumResult = gsSum(A, b, x0, nIter, threshold, checkeoNorma);
        end = chrono::high_resolution_clock::now();
        double gsSumTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
        double gsSumError = (expected - gsSumResult).norm();

        // Escribir los resultados en archivos
        writeResultToFile("results.csv", "LU", luTime, 0.0);
        writeResultToFile("results.csv", "jSum", jsumTime, jsumError);
        writeResultToFile("results.csv", "gsSum", gsSumTime, gsSumError);
    }

    return 0;
}
