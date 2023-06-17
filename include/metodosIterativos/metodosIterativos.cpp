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

VectorXd jMatIter(MatrixXd& A, VectorXd& b, VectorXd& x0, int nIter){
    MatrixXd L = strictlyLowerTriangularView(A) * (-1);
    MatrixXd U = strictlyUpperTriangularView(A) * (-1);
    MatrixXd D = A.diagonal().asDiagonal();
    
    MatrixXd invD = D.inverse();
    MatrixXd c = invD * b;
    MatrixXd R = invD * (L + U);
    
    VectorXd xk1, xk = x0;

    for(int k = 0; k < nIter; k++){
        xk1 = (R * xk) + c;
        xk = xk1;
    }

    return xk1;
}

VectorXd jMatThresh(MatrixXd& A, VectorXd& b, VectorXd& x0, float threshold){
    MatrixXd L = strictlyLowerTriangularView(A) * (-1);
    MatrixXd U = strictlyUpperTriangularView(A) * (-1);
    MatrixXd D = A.diagonal().asDiagonal();
    
    MatrixXd invD = D.inverse();
    MatrixXd c = invD * b;
    MatrixXd R = invD * (L + U);
    
    VectorXd xk1, xk = x0;
    
    while((xk1 - xk).norm() > threshold){
        xk1 = (R * xk) + c;
        xk = xk1;
    }

    return xk1;
}


VectorXd gsMatIter(MatrixXd& A, VectorXd& b, VectorXd& x0, int nIter){
    MatrixXd L = strictlyLowerTriangularView(A) * (-1);
    MatrixXd U = strictlyUpperTriangularView(A) * (-1);
    MatrixXd D = A.diagonal().asDiagonal();
    
    MatrixXd DLinv = (D - L).inverse();
    MatrixXd c = DLinv * b;
    MatrixXd R = DLinv * U; 
    
    VectorXd xk1, xk = x0;

    for(int k = 0; k < nIter; k++){
        xk1 = (R * xk) + c;
        xk = xk1;
    }

    return xk1;
}

double sumatoriaDeJIter(MatrixXd& M, int i, VectorXd& xk) {
    double sum = 0;
    int n = M.cols();
    for(int j = 0; j < n; j++) {
        if(j != i)
            sum += M.coeff(i, j) * xk.coeff(j);
    }
    return sum;
}

VectorXd jSumIter(MatrixXd& A, VectorXd& b, VectorXd& x0, int nIter) {
    int n = A.rows(); 
    VectorXd xk = x0;
    VectorXd xk1(xk.size());
    for(int k = 0; k < nIter; k++) {
        for(int i = 0; i < n; i++) {
            double sum = sumatoriaDeJ(A, i, xk);
            double b_i = b.coeff(i);
            double a_ii = A.coeff(i, i);
            double xk1_i = (b_i - sum) / a_ii;
            xk1(i) = xk1_i;
        }
        xk = xk1;
    }
    return xk1;
}

double sumatoriaDeGS1(MatrixXd& M, int i, VectorXd& xk) {
    double sum = 0;
    int n = M.cols();
    for (int j = i + 1; j < n; j++) {
        sum += M.coeff(i, j) * xk.coeff(j);
    }
    return sum;
}

double sumatoriaDeGS2(MatrixXd& M, int i, VectorXd& xk1) {
    double sum = 0;
    for (int j = 0; j < i; j++) {
        sum += M.coeff(i, j) * xk1.coeff(j);
    }
    return sum;
}

VectorXd gsSumIter(MatrixXd& A, VectorXd& b, VectorXd& x0, int nIter) {
    int n = A.rows();
    VectorXd xk = x0;
    VectorXd xk1(size(xk));
    for (int k = 0; k < nIter; k++) {
        for (int i = 0; i < n; i++) {
            double sum1 = sumatoriaDeGS1(A, i, xk);
            double sum2 = sumatoriaDeGS2(A, i, xk1);
            double b_i = b.coeff(i);
            double a_ii = A.coeff(i, i);
            double xk1_i = (b_i - sum1 - sum2) / a_ii;
            xk1(i) = xk1_i;
        }
        xk = xk1;
    }
    return xk1;
}

VectorXd resolverLU(MatrixXd& A, VectorXd& b){
    VectorXd x = A.lu().solve(b);
    return x; 
}


