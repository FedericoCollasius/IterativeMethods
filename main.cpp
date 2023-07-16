#include "include/metodosIterativos/metodosIterativos.h"
#include <chrono>
#include <cfloat>

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
        matrix(i, i) = sum + 1; 
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

void writeResultToFile2(string filename, int matrixSize, int nIter, vector<vector<double>>& tiempos, vector<vector<double>>& errores, int matrixType, double condNumber){
    ofstream file(filename, ios::app);
    if(file.is_open()){
        for(int i = 0; i < tiempos[0].size(); i++){
            file << matrixSize << "," << matrixType << "," << condNumber << "," << nIter;
            for(auto& timeList : tiempos){
                file << "," << timeList[i];
            }
            for(auto& errorList : errores){
                file << "," << errorList[i];
            }
            file << endl;
        }
        file.close();
    } else {
        cout << "Error al abrir el archivo: " << filename << endl;
    }
}

void generarMatricesYCalcularCondicion(vector<int>& matrixSizes, int num_matrices, vector<MatrixXd>& mejoresMatrices,vector<MatrixXd>& peoresMatrices){
    for(auto& matrixSize : matrixSizes){
        double best_cond = DBL_MAX;
        MatrixXd best_matrix;

        double worst_cond = -DBL_MAX; // initialize with negative maximum double
        MatrixXd worst_matrix;
        for(int i = 0; i < num_matrices; i++){
            //Generate randm matrix A 
            MatrixXd A = generateDominantMatrix(matrixSize);
            JacobiSVD<MatrixXd> svd(A);
            double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);

            if(cond < best_cond){
                best_cond = cond;
                best_matrix = A;
            }

            if(cond > worst_cond){
                worst_cond = cond;
                worst_matrix = A;
            }
        }
        mejoresMatrices.push_back(best_matrix);
        peoresMatrices.push_back(worst_matrix);
    }
}

void valoresNumeroCondicion(int numTests, vector<int>& nIters, vector<MatrixXd>& mejoresMatrices, vector<MatrixXd>& peoresMatrices, double threshold, int check, int divThreshold){
    string filename = "valoresNumeroCondicion.csv";
    ofstream file(filename, ios::app);
    if (file.is_open()) {
        file << "Tamaño de la matriz" << "," << 
        "Tipo de Matriz (0 = mejor, 1 = peor)" << "," <<
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

    vector<vector<MatrixXd>> matricesGroups = {mejoresMatrices, peoresMatrices};

    for (int groupId = 0; groupId < matricesGroups.size(); groupId++) {
        auto& matrixGroup = matricesGroups[groupId];
        int matrixType = (groupId == 0) ? 0 : 1;

        for(auto& A : matrixGroup){
            JacobiSVD<MatrixXd> svd(A);
            double condNumber = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
            for(auto& nIter : nIters){
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
                writeResultToFile2(filename, A.rows(), nIter, tiempos, errores, matrixType, condNumber);
            }
        }
    }
}

int main(){
  //Generate highLowConditionNumber
  vector<MatrixXd> mejoresMatrices;
  while(true){
    highLowConditionNumber(10);
  }
}

int main2() {
    // Parámetros de prueba
    int numTests = 10;
    
    double threshold = 0.0001;
    int check = 10; 
    int divThreshold = 5;

    int numMatrices = 100; 

    vector<int> nIters = {10, 25, 50, 75, 100, 200, 400};
    vector<int> matrixSizes = {10, 50, 100, 150, 250, 500, 750, 1000, 1500, 2000, 2500};
    
    //valoresCrudos(numTests, nIters, matrixSizes, threshold, check, divThreshold);
    
    vector<MatrixXd> mejoresMatrices; 
    vector<MatrixXd> peoresMatrices;
    generarMatricesYCalcularCondicion(matrixSizes, numMatrices, mejoresMatrices, peoresMatrices);

    valoresNumeroCondicion(numTests, nIters, mejoresMatrices, peoresMatrices, threshold, check, divThreshold);

    return 0;
}

vector<MatrixXd> highLowConditionNumber(int size){
  //Generate matrixes of one column replicated, so that columns are almost lineal dependent
  //Generate 10 
  for (int i = 0; i < size; i++) {
    for( int j = 0; j < size; j++){
      if (i == j) {
        A(i,j) = 1;
      } else {
        A(i,j) = 1/size;
      }
    })  
  }
  //print condition number
  JacobiSVD<MatrixXd> svd(A);
  double condNumber = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
  cout << "Cond number: " << condNumber << endl;
  return A;
}
