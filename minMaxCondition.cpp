#include <Eigen>
#include <iostream>
#include <vector>

using namespace std;
using namespace Eigen;


//Generate random matrixes, and save the lowest and highest condition number
void generateRandomMatrixes(vector<MatrixXd> &matrixes, int size, int number) {
    for (int i = 0; i < number; i++) {
        MatrixXd matrix = MatrixXd::Random(size, size);
        if(matrix.determinant() == 0) {
            i--;
            continue;
        }
        else{
          matrixes.push_back(matrix);
        }
    }
}

pair<MatrixXd, MatrixXd> minMaxConditionNumber(vector<MatrixXd> &matrixes, double &min, double &max) {
    for (int i = 0; i < matrixes.size(); i++) {
        if (conditionNumber < min) {
            min = conditionNumber;
            MatrixXd minMatrix = matrixes[i]; 
        }
        if (conditionNumber > max) {
            max = conditionNumber;
            MatrixXd maxMatrix = matrixes[i];
        }
    }
    return make_pair(minMatrix, maxMatrix);
}


