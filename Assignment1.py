def generateIdentityeMatrix(size):
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
        m = Matrix(n,n)
        for i in range(n):
            for j in range(n):
                m.Rows[i][j] = self.Rows[i][j]
        for i in range(n):
            for j in range(i+1,n):
                if(m.Rows[i][i]==0):
                    m.Rows[i][i] = 1e-20
                c = m.Rows[j][i]/(m.Rows[i][i])
                for k in range(n):
                    m.Rows[j][k] -= c*m.Rows[i][k]
        prod = 1.0
        for i in range(n):
            prod*=m.Rows[i][i]
        return abs(prod)
    
    def inverse(self):
        assert self.numofCols() == self.numOfRows(),"Given matrix is not a Square Matrix"
        assert self.FindDeterminant()!=0,"Determinant of the matrix is zero"
        n = self.numofCols()
        m = self.copyMatrix()
        m1 = generateIdentityeMatrix(n)
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

    def setMatrix(self):
        n = self.numofCols()
        for i in range(n):
            for j in range(n):
                k = int(input())
                self.Rows[i][j] = k

        

m = Matrix(3,3)
m.setMatrix()

m.printMatrix()
print()
m1=m.inverse()
m1.printMatrix()





 

    

        