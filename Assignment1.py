# Section 2
# Agam Shah	        AU2140024
# Paraj Bhatt 	        AU2140110
# Vinay Mungra	        AU2140120
# Bhargav Kargatiya 	AU2140121
# Bhavy Jhaveri	        AU2140168

import numpy as np
import matplotlib.pyplot as pt
from mpl_toolkits.mplot3d import Axes3D
import math
def generateIdentityMatrix(size):  # -->Will generate an identity matrix of whatever size you give as input
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
        self.Rows=[[None]*numCols for i in range(numRows)]  # 1. initializing matrix with all elements set as none
    
    def numOfRows(self):
        return len(self.Rows)                               # 2. Returns number of rows
    
    def numofCols(self):
        return len(self.Rows[0])                            # 3. returns number of columns

    def copyMatrix(self):                                   # 4. Creates a copy of the matrix
        n = self.numofCols()
        n1 = self.numOfRows()
        
        m = Matrix(n1,n)
        for i in range(n1):
            for j in range(n):
                m.Rows[i][j] = self.Rows[i][j]
        return m

    def setMatrix(self):                                   #-->We can set elements of the matrix manually
        n = self.numofCols()
        n1 = self.numOfRows()
        for i in range(n1):
            l1 = list(map(int,input().split()))
            for j in range(len(l1)):
                self.Rows[i][j] = l1[j]
    
    def printMatrix(self):
        m = np.array(self.Rows)
        print(m)

    
    def transpose(self):                                 #-->Returns the transpose of the matrix
        m1 = Matrix(self.numofCols(),self.numOfRows())
        for i in range(self.numOfRows()):
            for j in range(self.numofCols()):
                m1.Rows[j][i] = self.Rows[i][j]
        return m1

    def FindDeterminant(self):                          #-->Finds the determinant of the matrix
        assert self.numofCols() == self.numOfRows(),"Given matrix is not a Square Matrix"
        n = self.numOfRows()
        m = self.copyMatrix()
        count = 0
        for i in range(n):
            for j in range(i+1,n):
                if(m.Rows[i][i]==0):
                    c1=i
                    while(m.Rows[c1][i]==0):
                        c1+=1
                    if(c1!=i):
                        count+=1
                    m.Rows[c1],m.Rows[i] = m.Rows[i],m.Rows[c1]

                c = m.Rows[j][i]/(m.Rows[i][i])
                for k in range(n):
                    m.Rows[j][k] = m.Rows[j][k] - c*m.Rows[i][k]
            """
            the code uptil now converted the matrix into a upper triangular matrix.
            Now all we have to do is to multiply the diagonal elements. But if there have
            been any interchanging of the rows we also have to multiply -1 which is what the count
            variable is for.
            """
        prod = 1.0
        for i in range(n):
            prod*=m.Rows[i][i]
        return prod*(pow(-1,count))  
    
    def inverse(self):                                                                    # A. We find the inverse of the matrix using Gaussian elimination method
        assert self.numofCols() == self.numOfRows(),"Given matrix is not a Square Matrix" # 1. if The matrix is not a sqaure matrix we cannot find its inverse
        assert self.FindDeterminant()!=0,"Determinant of the matrix is zero"              # 2. if the determinant of the matrix is zero we cannot find its inverse.
        n = self.numofCols()
        m = self.copyMatrix()
        m1 = generateIdentityMatrix(n)                              # 3. First generate an idenetity matrix which is same size as that of the matrix we want to inveret
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
    """
    This code will convert a copy of the matrix into identity matrix and through those same operations,
    convert identity matrix into inverse of the matrix.
    """
    
    def rref(self):                             # F. Row reduction of matrix. 
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

    def norm(self):                                                 # E. Calculate the norm of the matrix     
        n = self.numOfRows()
        m = self.numofCols()
        assert m==1,"Number of columns should be one for a vector"  #In a vector the number of columns should only be one
        sum = 0
        for i in range(n):
            sum = sum + self.Rows[i][0]*self.Rows[i][0]             #Taking squares of each element.
        return math.sqrt(sum)                                       #Taking the square root of the sum to get the norm.



    def rank(self):                            #G. This method will give the rank  of the given matrix
        m=self.rref()                          #1. It calls the Row Reduced function to covert given matrix in row echelon form. 
        nrow=self.numOfRows()
        ncol=self.numofCols()
        rank=0
        a=0
        for i in range(0,ncol):                # 2. In row reduced matrix, it iterates through every column.
            flag=False
            b = np.array([])                   # 3. To check every element, it appends a present number to the b array, 
            for  j in range(0,nrow):           #and after the column end, it checks the size and elements of the 
                if(m.Rows[j][i]!= 0):          #array by which we can know if the column is pivot or not.
                    b=np.append(b,m.Rows[j][i])
                if(m.Rows[j][i]==1 and j>a and not flag):
                    a=j
                    flag=True
                  
            if(len(b)==1):
                if(b[0]==1):
                    rank+=1                    # 4. If a present column is a pivot, then we can increase the rank by 1.
            b={}   
        return rank
     
        
def plotVector2D(x,y):                                                # D. This funciton will plot x,y as a vector in 2d plane
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

def plotVector3D(data):                      #This function will plot x,y,z as a vector in 3d plane.
    fig = pt.figure()
    ax = pt.axes(projection='3d')
    ax.set_xlim([-1,10])
    ax.set_ylim(-10,10)
    ax.set_zlim([0,10])
    start = [0,0,0]
    ax.quiver(start[0],start[1],start[2],data[0],data[1],data[2])
    pt.show()

    
m = Matrix(3,3)

m.setMatrix()
m.printMatrix()
print()


n = m.norm()
print(n)







 

    

        
