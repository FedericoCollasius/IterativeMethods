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

void writeResultToFile(string filename, int matrixSize, vector<double> tiemposPromedio, vector<double> erroresPromedio) {
    ofstream file(filename, ios::app);
    if (file.is_open()) {
        file << matrixSize << ',' << tiemposPromedio[0] << ',' << erroresPromedio[0] << ',' << tiemposPromedio[1] << ',' << erroresPromedio[1] << ',' << tiemposPromedio[2] << ',' << erroresPromedio[2] << ',' << tiemposPromedio[3] << ',' << erroresPromedio[3] << ',' << tiemposPromedio[4] << ',' << erroresPromedio[4] << endl;
        file.close();
    } else {
        cout << "Error al abrir el archivo: " << filename << endl;
    }
}

void writeResultToFile2(const std::string& filename, const vector<vector<double>>& tiempos, const vector<vector<double>>& errores) {
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        for (int i = 0; i < tiempos[0].size(); i++) { // para cada prueba individual
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


int main() {
    // Parámetros de prueba
    int numTests = 100;
    
    int nIter = 500;
    double threshold = 0.0001;
    int checkeoNorma = nIter - 1 / 10;
    int divThreshold = 5;
    double delta = 10;

    ofstream file("promedios.csv", ios::app);
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

    vector<int> values = {10};

    for (int i = 0; i < values.size(); i++) {
        int tamanioMatriz = values[i];
        vector<double> tiemposPromedio(5, 0.0);
        vector<double> erroresPromedio(5, 0.0);

        string filename = "valoresCrudos" + std::to_string(tamanioMatriz) + ".csv";
        ofstream file(filename, ios::app);
        if (file.is_open()) {
        file << "Tiempos LU" << "," << "Errores LU" << "," << 
        "Tiempos JMat" << "," << "Errores JMat" << "," << 
        "Tiempos GSMat" << "," << "Errores GSMat" << "," <<
        "Tiempos JSum" << "," << "Errores JSum" << "," <<
        "Tiempos GSSum" << "," << "Errores GSSum" << endl;
        file.close();
        } else {
        cout << "Error al abrir el archivo: " << filename << endl;
        }
        vector<vector<double>> tiempos(5, vector<double>(numTests, 0.0));
        vector<vector<double>> errores(5, vector<double>(numTests, 0.0));

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
            tiemposPromedio[0] += luTime;
            erroresPromedio[0] += luError;
            tiempos[0][i] = luTime;
            errores[0][i] = luError;

            start = chrono::high_resolution_clock::now();
            VectorXd jMatResult = jMat(A, b, x0, nIter, threshold, checkeoNorma, divThreshold);
            end = chrono::high_resolution_clock::now();
            double jMatTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
            double jMatError = (expected - jMatResult).norm();
            tiemposPromedio[1] += jMatTime;
            erroresPromedio[1] += jMatError;
            tiempos[1][i] = jMatTime;
            errores[1][i] = jMatError;

            start = chrono::high_resolution_clock::now();
            VectorXd gsMatResult = gsMat(A, b, x0, nIter, threshold, checkeoNorma, divThreshold);
            end = chrono::high_resolution_clock::now();
            double gsMatTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
            double gsMatError = (expected - gsMatResult).norm();
            tiemposPromedio[2] += gsMatTime;
            erroresPromedio[2] += gsMatError;
            tiempos[2][i] = gsMatTime;
            errores[2][i] = gsMatError;

            start = chrono::high_resolution_clock::now();
            VectorXd jSumResult = jSum(A, b, x0, nIter, threshold, checkeoNorma, divThreshold);
            end = chrono::high_resolution_clock::now();
            double jSumTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
            double jSumError = (expected - jSumResult).norm();
            tiemposPromedio[3] += jSumTime;
            erroresPromedio[3] += jSumError;
            tiempos[3][i] = jSumTime;
            errores[3][i] = jSumError;

            start = chrono::high_resolution_clock::now();
            VectorXd gsSumResult = gsSum(A, b, x0, nIter, threshold, checkeoNorma, divThreshold);
            end = chrono::high_resolution_clock::now();
            double gsSumTime = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
            double gsSumError = (expected - gsSumResult).norm();
            tiemposPromedio[4] += gsSumTime;
            erroresPromedio[4] += gsSumError;
            tiempos[4][i] = gsSumTime;
            errores[4][i] = gsSumError;
        }
        for(int i = 0; i < 5; i++){
            tiemposPromedio[i] /= numTests;
            erroresPromedio[i] /= numTests;
        }
        writeResultToFile("promedios.csv", tamanioMatriz, tiemposPromedio, erroresPromedio);
        writeResultToFile2(filename, tiempos, errores);
    }
    return 0;
}
