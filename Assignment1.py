import numpy as np
import matplotlib.pyplot as pt
from mpl_toolkits.mplot3d import Axes3D
import math
def generateIdentityMatrix(size):
        m = Matrix(size,size)
        for i in range(size):
            for j in range(size):
                if(i==j):
                    m.Rows[i][j] = 1
                else:
                    m.Rows[i][j] = 0
                    
        return m
class Matrix():
    def __init__(self,numRows,numCols):
        self.Rows=[[None]*numCols for i in range(numRows)]
    
    def numOfRows(self):
        return len(self.Rows)
    
    def numofCols(self):
        return len(self.Rows[0])

    def copyMatrix(self):
        n = self.numofCols()
        n1 = self.numOfRows()
        
        m = Matrix(n1,n)
        for i in range(n1):
            for j in range(n):
                m.Rows[i][j] = self.Rows[i][j]
        return m

    def setMatrix(self):
        n = self.numofCols()
        n1 = self.numOfRows()
        for i in range(n1):
            for j in range(n):
                k = int(input())
                self.Rows[i][j] = k
    
    def printMatrix(self):
        for i in range(self.numOfRows()):
            for j in range(self.numofCols()):
                print(self.Rows[i][j],end=" ")
            print()
    
    def transpose(self):
        m1 = Matrix(self.numofCols(),self.numOfRows())
        for i in range(self.numOfRows()):
            for j in range(self.numofCols()):
                m1.Rows[j][i] = self.Rows[i][j]
        return m1

    def FindDeterminant(self):
        assert self.numofCols() == self.numOfRows(),"Given matrix is not a Square Matrix"
        n = self.numOfRows()
        m = self.copyMatrix()
        for i in range(n):
            for j in range(i+1,n):
                if(m.Rows[i][i]==0):
                    c1=i
                    while(m.Rows[c1][i]==0):
                        c1+=1
                    m.Rows[c1],m.Rows[i] = m.Rows[i],m.Rows[c1]

                c = m.Rows[j][i]/(m.Rows[i][i])
                for k in range(n):
                    m.Rows[j][k] = m.Rows[j][k] - c*m.Rows[i][k]
        prod = 1.0
        for i in range(n):
            prod*=m.Rows[i][i]
        return prod
    
    def inverse(self):
        assert self.numofCols() == self.numOfRows(),"Given matrix is not a Square Matrix"
        assert self.FindDeterminant()!=0,"Determinant of the matrix is zero"
        n = self.numofCols()
        m = self.copyMatrix()
        m1 = generateIdentityMatrix(n)
        pivot = 0
        for row in range(n):
            if(n<=pivot):
                break
            i = row
            while(m.Rows[i][pivot]==0):
                i+=1
                if(i==row):
                    pivot+=1
                if(n==pivot):
                    break
            m.Rows[i],m.Rows[row] = m.Rows[row],m.Rows[i]
            m1.Rows[i],m1.Rows[row] = m1.Rows[row],m1.Rows[i]
            c = m.Rows[row][pivot]
            for j in range(n):
                m.Rows[row][j] = m.Rows[row][j]/c
                m1.Rows[row][j] = m1.Rows[row][j]/c
                
            for k in range(n):
                if(k!=row):
                    c1 = m.Rows[k][pivot]
                    for j in range(n):
                        m.Rows[k][j] = m.Rows[k][j] - c1*m.Rows[pivot][j]
                        m1.Rows[k][j] = m1.Rows[k][j] - c1*m1.Rows[pivot][j]
            pivot+=1
        return m1

    def rref(self):
        N = self.numOfRows()
        M = self.numofCols()
        m = self.copyMatrix()
        pivot = 0
        for row in range(N):
            if(M<=pivot):
                return m
            i = row
            while(m.Rows[i][pivot]==0):
                i+=1
                if(i==N):
                    i = row
                    pivot+=1
                    if(pivot==M):
                        return m
            m.Rows[i],m.Rows[row] = m.Rows[row],m.Rows[i]
            c = m.Rows[row][pivot]
            if(c!=0):
                for j in range(M):
                    m.Rows[row][j] = m.Rows[row][j]/c
            
            for k in range(N):
                if(k!=row):
                    c1 = m.Rows[k][pivot]
                    for j in range(M):
                        m.Rows[k][j] = m.Rows[k][j] - c1*m.Rows[row][j]
            pivot+=1
        return m     

    def norm(self):
        n = self.numOfRows()
        m = self.numofCols()
        assert m==1,"Number of columns should be one for a vector"
        sum = 0
        for i in range(n):
            sum = sum + self.Rows[i][0]*self.Rows[i][0]
        return math.sqrt(sum)


def plotVector2D(x,y):
    pt.rcParams["figure.figsize"] = [10,10]
    pt.rcParams["figure.autolayout"] = True
    data = np.array([0,0,x,y])
    pt.figure() #creates new figure
    ax = pt.gca() #gets current axis
    ax.quiver(*data,angles='xy',scale_units='xy',scale=1,color='blue') #we can style how the arrow looks using this
    ax.set_xlim([-1,10])
    ax.set_ylim([-1,10])
    pt.draw()
    pt.show()

def plotVector3D(data):
    fig = pt.figure()
    ax = pt.axes(projection='3d')
    ax.set_xlim([-1,10])
    ax.set_ylim(-10,10)
    ax.set_zlim([0,10])
    start = [0,0,0]
    ax.quiver(start[0],start[1],start[2],data[0],data[1],data[2])
    pt.show()

    
m = Matrix(3,1)

m.setMatrix()
m.printMatrix()
print()


n = m.norm()
print(n)







 

    

        