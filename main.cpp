#include <iostream>
#include <Eigen/Dense>
#include "include/metodosIterativos/metodosIterativos.h"
using namespace std;
using namespace Eigen;


int main() {   
    MatrixXd matrix(3, 3);
    matrix << 1, -0.5, 0.5,
               1, 1, 1, 
               -0.5, -0.5, 1;

    VectorXd b(3);
    b << 4, -1, 1;           
    
    VectorXd x0(3); 
    x0 << 0, 0, 0;

    int nIter = 50; 

    VectorXd jacobiIter = jMatIter(matrix, b, x0, nIter);
    VectorXd gsIter = gsMatIter(matrix, b, x0, nIter);
    VectorXd expected = resolverLU(matrix, b);

    return 0; 
}
