#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <cmath>

using namespace std;
using namespace Eigen;

VectorXd jMat(MatrixXd& A, VectorXd& b, VectorXd& x0, int nIter);
VectorXd jSum(MatrixXd& A, VectorXd& b, VectorXd& x0, int nIter);

VectorXd gsMat(MatrixXd& A, VectorXd& b, VectorXd& x0, int nIter);
VectorXd gsSum(MatrixXd& A, VectorXd& b, VectorXd& x0, int nIter);

MatrixXd strictlyLowerTriangularView(MatrixXd& M);
MatrixXd strictlyUpperTriangularView(MatrixXd& M);


