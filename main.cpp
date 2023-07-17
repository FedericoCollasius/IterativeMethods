#include "include/metodosIterativos/metodosIterativos.h"
#include <vector>
#include <float.h>
#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>

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
                matrix(i, j) = rand() % 10;
                sum += matrix(i, j);
            }
        }
        matrix(i, i) = sum + 1; 
    }
    return matrix;
}

MatrixXd generateDiagonllyDominantLowCond(int size){
    MatrixXd matrix(size, size);
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            if(i == j){
                matrix(i, j) = 1;
            }
            else{
                matrix(i, j) = 1/size;
            }
        }
    }
    return matrix; 
}

VectorXd generateRandomVector(int size) {
    VectorXd vector(size);
    int random_int = rand() % 10;
    vector = VectorXd::Constant(size, random_int);
    return vector;
}

void writeResultToFile(const std::string& filename, int matrixSize, int nIter, const vector<vector<double>>& tiempos, const vector<vector<double>>& errores) {
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        for (int i = 0; i < tiempos[0].size(); i++) { // para cada prueba individual
            file << matrixSize << "," << nIter << ",";
            for (int j = 0; j < tiempos.size(); j++) { // para cada método
                file << tiempos[j][i] << ",";
                file << errores[j][i];
                if (j != tiempos.size() - 1) {
                    file << ",";
                }
            }
            file << std::endl;
        }
        file.close();
    } else {
        std::cout << "Error al abrir el archivo: " << filename << std::endl;
    }
}

void valoresCrudos(int numTests, vector<int>& nIters, vector<int>& matrixSizes, double threshold, int check, int divThreshold){
    string filename = "valoresCrudos.csv";
    ofstream file(filename, ios::app);
    if (file.is_open()) {
        file << "Tamaño de la matriz" << "," << "Número de iteraciones" << "," << 
        "Tiempos LU" << "," << "Errores LU" << "," << 
        "Tiempos JMat" << "," << "Errores JMat" << "," << 
        "Tiempos GSMat" << "," << "Errores GSMat" << "," <<
        "Tiempos JSum" << "," << "Errores JSum" << "," <<
        "Tiempos GSSum" << "," << "Errores GSSum" << endl;
        file.close();
    } else {
        cout << "Error al abrir el archivo: " << filename << endl;
    }

    for (auto& matrixSize : matrixSizes) {
        for (auto& nIter : nIters) {
            vector<vector<double>> tiempos(5, vector<double>(numTests, 0.0));
            vector<vector<double>> errores(5, vector<double>(numTests, 0.0));

            for (int i = 0; i < numTests; i++) {
                MatrixXd A = generateDominantMatrix(matrixSize);
                VectorXd b = generateRandomVector(matrixSize); 
                // Como Ax = b  => x = A^-1 * b
                VectorXd expected = A.inverse() * b;
                VectorXd x0 = generateRandomVector(matrixSize);

                auto start = chrono::high_resolution_clock::now();
                VectorXd luResult = resolverLU(A, b);
                auto end = chrono::high_resolution_clock::now();
                double luTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
                double luError = (expected - luResult).norm();
                tiempos[0][i] = luTime;
                errores[0][i] = luError;

                start = chrono::high_resolution_clock::now();
                VectorXd jMatResult = jMat(A, b, x0, nIter, threshold, check, divThreshold);
                end = chrono::high_resolution_clock::now();
                double jMatTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
                double jMatError = (expected - jMatResult).norm();
                tiempos[1][i] = jMatTime;
                errores[1][i] = jMatError;

                start = chrono::high_resolution_clock::now();
                VectorXd gsMatResult = gsMat(A, b, x0, nIter, threshold, check, divThreshold);
                end = chrono::high_resolution_clock::now();
                double gsMatTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
                double gsMatError = (expected - gsMatResult).norm();
                tiempos[2][i] = gsMatTime;
                errores[2][i] = gsMatError;

                start = chrono::high_resolution_clock::now();
                VectorXd jSumResult = jSum(A, b, x0, nIter, threshold, check, divThreshold);
                end = chrono::high_resolution_clock::now();
                double jSumTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
                double jSumError = (expected - jSumResult).norm();
                tiempos[3][i] = jSumTime;
                errores[3][i] = jSumError;

                start = chrono::high_resolution_clock::now();
                VectorXd gsSumResult = gsSum(A, b, x0, nIter, threshold, check, divThreshold);
                end = chrono::high_resolution_clock::now();
                double gsSumTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
                double gsSumError = (expected - gsSumResult).norm();
                tiempos[4][i] = gsSumTime;
                errores[4][i] = gsSumError;
            }
            writeResultToFile(filename, matrixSize, nIter, tiempos, errores);
        }
    }
}


void writeResultToFile2(string filename, int matrixSize, int nIter, vector<vector<double>>& tiempos, 
vector<vector<double>>& errores, double condNumber){
    ofstream file(filename, ios::app);
    if(file.is_open()){
        for(int i = 0; i < tiempos[0].size(); i++){
            file << matrixSize << "," << condNumber << "," << nIter << ",";
            for (int j = 0; j < tiempos.size(); j++) { // para cada método
                file << tiempos[j][i] << ",";
                file << errores[j][i];
                if (j != tiempos.size() - 1) {
                    file << ",";
                }
            }
            file << std::endl;
        }
        file.close();
    } else {
        cout << "Error al abrir el archivo: " << filename << endl;
    }
}

void valoresNumeroCondicion(int numTests, int iters, 
    vector<pair<double, MatrixXd>>& matrices, 
    double threshold, int check, int divThreshold){

    string filename = "valoresNumeroCondicion.csv";
    ofstream file(filename, ios::app);
    if (file.is_open()) {
        file << "Tamaño de la matriz" << "," << 
        "Número de Condición" << "," <<
        "Número de iteraciones" << "," << 
        "Tiempos LU" << "," << "Errores LU" << "," << 
        "Tiempos JMat" << "," << "Errores JMat" << "," << 
        "Tiempos GSMat" << "," << "Errores GSMat" << "," <<
        "Tiempos JSum" << "," << "Errores JSum" << "," <<
        "Tiempos GSSum" << "," << "Errores GSSum" << endl;
        file.close();
    } else {
        cout << "Error al abrir el archivo: " << filename << endl;
    }

    for(auto& matrixPair: matrices){
        double condNumber = matrixPair.first;
        MatrixXd A = matrixPair.second;
        vector<vector<double>> tiempos(5, vector<double>(numTests, 0.0));
        vector<vector<double>> errores(5, vector<double>(numTests, 0.0));

        for(int i = 0; i < numTests; i++){
            VectorXd b = generateRandomVector(A.rows());
            VectorXd expected = A.inverse() * b;
            VectorXd x0 = generateRandomVector(A.rows());

            auto start = chrono::high_resolution_clock::now();
            VectorXd luResult = resolverLU(A, b);
            auto end = chrono::high_resolution_clock::now();
            double luTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
            double luError = (expected - luResult).norm();
            tiempos[0][i] = luTime;
            errores[0][i] = luError;

            start = chrono::high_resolution_clock::now();
            VectorXd jMatResult = jMat(A, b, x0, iters, threshold, check, divThreshold);
            end = chrono::high_resolution_clock::now();
            double jMatTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
            double jMatError = (expected - jMatResult).norm();
            tiempos[1][i] = jMatTime;
            errores[1][i] = jMatError;

            start = chrono::high_resolution_clock::now();
            VectorXd gsMatResult = gsMat(A, b, x0, iters, threshold, check, divThreshold);
            end = chrono::high_resolution_clock::now();
            double gsMatTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
            double gsMatError = (expected - gsMatResult).norm();
            tiempos[2][i] = gsMatTime;
            errores[2][i] = gsMatError;

            start = chrono::high_resolution_clock::now();
            VectorXd jSumResult = jSum(A, b, x0, iters, threshold, check, divThreshold);
            end = chrono::high_resolution_clock::now();
            double jSumTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
            double jSumError = (expected - jSumResult).norm();
            tiempos[3][i] = jSumTime;
            errores[3][i] = jSumError;

            start = chrono::high_resolution_clock::now();
            VectorXd gsSumResult = gsSum(A, b, x0, iters, threshold, check, divThreshold);
            end = chrono::high_resolution_clock::now();
            double gsSumTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
            double gsSumError = (expected - gsSumResult).norm();
            tiempos[4][i] = gsSumTime;
            errores[4][i] = gsSumError;
        }
        writeResultToFile2(filename, A.rows(), iters, tiempos, errores, condNumber);
    }
}

void generateRandomMatrixes(vector<pair<double, MatrixXd>> &matrixes, int size, int number) {
    for (int i = 0; i < number; i++) {
        MatrixXd matrix = MatrixXd::Random(size, size);
        if(abs(matrix.determinant()) < 0.02) {
            i--;
            cout << "Matriz no invertible" << endl;
        }
        else{
          double condNumber = matrix.norm() * matrix.inverse().norm();
          matrixes.push_back(make_pair(condNumber, matrix));
        }
    }
    cout << "Matrices generadas de tamaño: " << size << endl;
}


int main() {
    cout << "Empezando..." << endl;
    
    int numTests = 1;
    double threshold = 0.0001;
    int check = 10; 
    int divThreshold = 5;

    vector<int> nIters = {10};
    vector<int> matrixSizes = {500};
    //valoresCrudos(numTests, nIters, matrixSizes, threshold, check, divThreshold);
    
    int numMatrices = 10000;
    int size = 16;
    int nIter = 25; 
    vector<pair<double, MatrixXd>> matrixes;
    generateRandomMatrixes(matrixes, size, numMatrices);
    valoresNumeroCondicion(numTests, nIter, matrixes, threshold, check, divThreshold);

    return 0;
}

