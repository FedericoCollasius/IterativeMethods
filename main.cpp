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

MatrixXd generateMatrixWithConditionNumber(int size, double conditionNumber) {
    MatrixXd matrix;
    double cond = 0;
    do {
        matrix = generateDominantMatrix(size);
        JacobiSVD<MatrixXd> svd(matrix);
        double maxSingularValue = svd.singularValues()(0);
        double minSingularValue = svd.singularValues()(svd.singularValues().size()-1);
        cond = maxSingularValue / minSingularValue;
    } while (abs(cond - conditionNumber) > 1e-3);
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

void writeResultToFile2(const std::string& filename, int matrixSize, int nIter, const vector<vector<double>>& tiempos, const vector<vector<double>>& errores, const vector<double>& condNumbers) {
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        for (int i = 0; i < tiempos[0].size(); i++) { // para cada prueba individual
            file << matrixSize << "," << nIter << ",";
            file << condNumbers[i] << ",";
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

void variacionErrorNumeroCondicion(int numTests, vector<int>& nIters, vector<int>& matrixSizes, double threshold, int check, int divThreshold){
    string filename = "valoresCrudosNumCondicion.csv";
    ofstream file(filename, ios::app);
    if (file.is_open()) {
        file << "Tamaño de la matriz" << "," << "Número de iteraciones" << "," << 
        "Número de condicion" << "," <<
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
            vector<double> condNumbers(numTests, 0.0);

            for (int i = 0; i < numTests; i++) {
                double conditionNumber = (i + 1) * 10.0; 
                MatrixXd A = generateMatrixWithConditionNumber(matrixSize, conditionNumber);
                VectorXd b = generateRandomVector(matrixSize);
                VectorXd expected = A.inverse() * b;
                VectorXd x0 = generateRandomVector(matrixSize);

                condNumbers[i] = conditionNumber;

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

            writeResultToFile2(filename, matrixSize, nIter, tiempos, errores, condNumbers);
        }
    }
}


int main() {
    // Parámetros de prueba
    int numTests = 100;
    
    double threshold = 0.0001;
    int check = 10; 
    int divThreshold = 5;

    vector<int> nIters = {10, 25, 50, 75, 100, 200, 400, 500};
    vector<int> matrixSizes = {10, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500};
    
    valoresCrudos(numTests, nIters, matrixSizes, threshold, check, divThreshold); 
    variacionErrorNumeroCondicion(numTests, nIters, matrixSizes, threshold, check, divThreshold);

    return 0; 
}

