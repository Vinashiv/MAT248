

class Matrix():
    def __init__(self,numRows,numCols):
        self.Rows=[[None]*numCols for i in range(numRows)]
    
    def numOfRows(self):
        return len(self.Rows)
    
    def numofCols(self):
        return len(self.Rows[0])
    
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

    

m = Matrix(3,3)
c=1
for i in range(3):
    for j in range(3):
        m.Rows[i][j] = c
        c+=1

m.printMatrix()

det = m.FindDeterminant()
print(det)

 

    

        