#include "./metodosIterativos.h"

MatrixXd strictlyLowerTriangularView(MatrixXd& M){
    MatrixXd  L = M.triangularView<Lower>();
    for(int i = 0; i < M.rows(); i++)
        L(i, i) = 0;
    return L;
}

MatrixXd strictlyUpperTriangularView(MatrixXd& M){
    MatrixXd U = M.triangularView<Upper>();
    for(int i = 0; i < M.rows(); i++)
        U(i, i) = 0;
    return U;
}

VectorXd jMat(MatrixXd& A, VectorXd& b, VectorXd& x0, int nIter){
    MatrixXd L = strictlyLowerTriangularView(A) * (-1);
    MatrixXd U = strictlyUpperTriangularView(A) * (-1);
    MatrixXd D = A.diagonal().asDiagonal();
    
    MatrixXd invD = D.inverse();
    MatrixXd c = invD * b;
    MatrixXd R = invD * (L + U);
    
    VectorXd xk1, xk = x0;

    for(int i = 0; i < nIter; i++){
        xk1 = (R * xk) + c;
        xk = xk1;
    }

    return xk1;
}


VectorXd jSum(MatrixXd& A, VectorXd& b, VectorXd& x0, int nIter){

}

VectorXd gsMat(MatrixXd& A, VectorXd& b, VectorXd& x0, int nIter){
    MatrixXd L = strictlyLowerTriangularView(A) * (-1);
    MatrixXd U = strictlyUpperTriangularView(A) * (-1);
    MatrixXd D = A.diagonal().asDiagonal();
    
    MatrixXd DLinv = (D - L).inverse();
    MatrixXd c = DLinv * b;
    MatrixXd R = DLinv * U; 
    
    VectorXd xk1, xk = x0;

    for(int i = 0; i < nIter; i++){
        xk1 = (R * xk) + c;
        xk = xk1;
    }

    return xk1;
}
VectorXd gsSum(MatrixXd& A, VectorXd& b, VectorXd& x0, int nIter){
    
}