#include <iostream>
#include <Eigen>
#include "include/metodosIterativos/metodosIterativos.h"

using namespace std;
using namespace Eigen;

int main() {   
  
    MatrixXd A(3, 3);
    A << 10, -1, 2,
         -1, 11, -1,
         2, -1, 10;
    
    VectorXd b(3);
    b << 4, -1, 1;           
    
    VectorXd x0(3); 
    x0 << 0, 0, 0;
    
    int nIter = 50; 

    VectorXd expected = resolverLU(A, b);
    VectorXd jacobi = jMat(A, b, x0, nIter);
    
    cout << "Difference between expected and Jacobi method: " << (expected - jacobi).norm() << endl;
    
    return 0; 
}
