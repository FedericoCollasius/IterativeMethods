#include "metodosIterativos.h"
#include <cmath> // Incluir la biblioteca <cmath> para std::isnan()


MatrixXd estrictamenteTriangularInferior(MatrixXd& M){
    MatrixXd  L = M.triangularView<Lower>();
    for(int i = 0; i < M.rows(); i++)
        L(i, i) = 0;
    return L;
}

MatrixXd estrictamenteTriangularSuperior(MatrixXd& M){
    MatrixXd U = M.triangularView<Upper>();
    for(int i = 0; i < M.rows(); i++)
        U(i, i) = 0;
    return U;
}

VectorXd jMat(MatrixXd& A, VectorXd& b, VectorXd& x0, int nIter, double threshold, int check, int divThreshold){
    MatrixXd L = estrictamenteTriangularInferior(A) * (-1);
    MatrixXd U = estrictamenteTriangularSuperior(A) * (-1);
    MatrixXd D = A.diagonal().asDiagonal();
    
    MatrixXd invD = D.inverse();
    MatrixXd c = invD * b;
    MatrixXd R = invD * (L + U);
    
    VectorXd xk1, xk = x0;
    double normaPrevia = xk.norm();
    int divChecker = 0;

    for(int k = 0; k < nIter; k++){
        xk1 = (R * xk) + c;
        
        // Verificar NaN
        if (xk1.array().isNaN().any()) {
            cout << "ALERTA: Se encontraron valores NaN en la iteración de jMat" << k << endl;
            break;
        }

        if ((xk1 - xk).norm() < threshold)
            break;

        if (k % check == 0) {
            double normaActual = xk1.norm();
            if (normaActual > normaPrevia) {
                divChecker++; 
                if(divChecker > divThreshold) {
                    cout << "ALERTA: El Metodo parece estar divirgiendo." << endl;
                    break;
                }
            }
            else{
                divChecker = 0;
            }
            normaPrevia = normaActual;
        }

        xk = xk1;
    }
    return xk1;
}

VectorXd gsMat(MatrixXd& A, VectorXd& b, VectorXd& x0, int nIter, double threshold, int check, int divThreshold){
    MatrixXd L = estrictamenteTriangularInferior(A) * (-1);
    MatrixXd U = estrictamenteTriangularSuperior(A) * (-1);
    MatrixXd D = A.diagonal().asDiagonal();
    
    MatrixXd DLinv = (D - L).inverse();
    MatrixXd c = DLinv * b;
    MatrixXd R = DLinv * U; 
    
    VectorXd xk1, xk = x0;
    double normaPrevia = xk.norm();
    int divChecker = 0;

    for(int k = 0; k < nIter; k++){
        xk1 = (R * xk) + c;
        
        // Verificar NaN
        if (xk1.array().isNaN().any()) {
            cout << "ALERTA: Se encontraron valores NaN en la iteración de gsMat " << k << endl;
            break;
        }

        if ((xk1 - xk).norm() < threshold)
            break;
        if (k % check == 0) {
            double normaActual = xk1.norm();
            if (normaActual > normaPrevia) {
                divChecker++; 
                if(divChecker > divThreshold) {
                    cout << "ALERTA: El Metodo parece estar divirgiendo." << endl;
                    break;
                }
            }
            else{
                divChecker = 0;
            }
            normaPrevia = normaActual;
        }

        xk = xk1;
    }
    return xk1;
}

double sumatoriaDeJ(MatrixXd& M, int i, VectorXd& xk) {
    double sum = 0;
    int n = M.cols();
    for(int j = 0; j < n; j++) {
        if(j != i)
            sum += M.coeff(i, j) * xk.coeff(j);
    }
    return sum;
}

VectorXd jSum(MatrixXd& A, VectorXd& b, VectorXd& x0, int nIter, double threshold, int check, int divThreshold) {
    int n = A.rows(); 
    VectorXd xk = x0;
    VectorXd xk1(xk.size());
    double normaPrevia = xk.norm();
    int divChecker = 0;

    for(int k = 0; k < nIter; k++) {
        for(int i = 0; i < n; i++) {
            double sum = sumatoriaDeJ(A, i, xk);
            double b_i = b.coeff(i);
            double a_ii = A.coeff(i, i);
            double xk1_i = (b_i - sum) / a_ii;
            xk1(i) = xk1_i;
        }

        // Verificar NaN
        if (xk1.array().isNaN().any()) {
            cout << "ALERTA: Se encontraron valores NaN en la iteración de jSum " << k << endl;
            break;
        }

        if ((xk1 - xk).norm() < threshold)
            break;
        if (k % check == 0) {
            double normaActual = xk1.norm();
            if (normaActual > normaPrevia) {
                divChecker++; 
                if(divChecker > divThreshold) {
                    cout << "ALERTA: El Metodo parece estar divirgiendo." << endl;
                    break;
                }
            }
            else{
                divChecker = 0;
            }
            normaPrevia = normaActual;
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

VectorXd gsSum(MatrixXd& A, VectorXd& b, VectorXd& x0, int nIter, double threshold, int check, int divThreshold) {
    int n = A.rows();
    VectorXd xk = x0;
    VectorXd xk1(xk.size());
    double normaPrevia = xk.norm();
    int divChecker = 0;

    for (int k = 0; k < nIter; k++) {
        for (int i = 0; i < n; i++) {
            double sum1 = sumatoriaDeGS1(A, i, xk);
            double sum2 = sumatoriaDeGS2(A, i, xk1);
            double b_i = b.coeff(i);
            double a_ii = A.coeff(i, i);
            double xk1_i = (b_i - sum1 - sum2) / a_ii;
            xk1(i) = xk1_i;
        }

        // Verificar NaN
        if (xk1.array().isNaN().any()) {
            cout << "ALERTA: Se encontraron valores NaN en la iteración de gsSum " << k << endl;
            break;
        }

        double normaActual = xk1.norm();
        if (k % check == 0) {
            double normaActual = xk1.norm();
            if (normaActual > normaPrevia) {
                divChecker++; 
                if(divChecker > divThreshold) {
                    cout << "ALERTA: El Metodo parece estar divirgiendo." << endl;
                    break;
                }
            }
            else{
                divChecker = 0;
            }
            normaPrevia = normaActual;
        }

        xk = xk1;
    }
    return xk1;
}

VectorXd resolverLU(MatrixXd& A, VectorXd& b){
    VectorXd x = A.lu().solve(b);
    return x; 
}
