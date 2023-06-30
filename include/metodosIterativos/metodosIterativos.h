#include <Eigen/Dense>
#include <utility>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>

using namespace std;
using namespace Eigen;

// Funciones Auxiliares: 
MatrixXd estrictamenteTriangularInferior(MatrixXd& M);
MatrixXd estrictamenteTriangularSuperior(MatrixXd& M);
double sumatoriaDeJ(MatrixXd& M, int i, VectorXd& xk);
double sumatoriaDeGS1(MatrixXd& M, int i, VectorXd& xk);
double sumatoriaDeGS2(MatrixXd& M, int i, VectorXd& xk1);
VectorXd resolverLU(MatrixXd& A, VectorXd& b);

// Metodos Iterativos:
VectorXd jMat(MatrixXd& A, VectorXd& b, VectorXd& x0, int nIter, double threshold, int checkeoNorma, int divThreshold, double delta);
VectorXd jSum(MatrixXd& A, VectorXd& b, VectorXd& x0, int nIter, double threshold, int checkeoNorma, int divThreshold, double);
VectorXd gsMat(MatrixXd& A, VectorXd& b, VectorXd& x0, int nIter, double threshold, int checkeoNorma, int divThreshold, double delta);
VectorXd gsSum(MatrixXd& A, VectorXd& b, VectorXd& x0, int nIter, double threshold, int checkeoNorma, int divThreshold, double delta);




