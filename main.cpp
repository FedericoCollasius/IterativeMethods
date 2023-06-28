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
    double threshold = 0.00001;

    VectorXd expected = resolverLU(A, b);
    VectorXd jacobi_mat = jSum(A, b, x0, nIter, threshold);
    
    cout << "Diferencia entre Metodo Directo y Metodo Iterativo: " << (expected - jacobi_mat).norm() << endl;

    return 0; 
}
