#Script to generate easily solvable matrices for testing Jacobi/Gauss-Seidel
#Ax=b 

import numpy as np
import sys


def generateMatrix(n):
    # Generate diagonal non-zero and non-singular nxn matrix
    A = np.zeros((n,n))
    for i in range(0,n):
        A[i][i] = np.random.randint(1,10)
    return A

def generateVector(n):
    #Generate a random nx1 vector with random values between 0 and 100
    b = np.random.randint(1,5,(n,1))
    return b

def generateSolution(A,b):
    #Solve the system Ax=b for x
    x = np.linalg.solve(A,b)
    return x

def main():
    #Create matrixes and save them to files
    for i in range(1,11):
        A = generateMatrix(i)
        b = generateVector(i)
        x = generateSolution(A,b)
        np.savetxt('A' + str(i) + '.txt', A, fmt='%d')
        np.savetxt('b' + str(i) + '.txt', b, fmt='%d')
        np.savetxt('x' + str(i) + '.txt', x, fmt='%d')

if __name__ == "__main__":
    main()
