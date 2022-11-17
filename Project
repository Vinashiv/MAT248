import numpy as np
from numpy.linalg import eig
import math



def largestElement(a, k, l):
  largestElement = 0
  v = [k,l,largestElement]
  largestElement = 0
  for i in range(rows):
    for j in range(cols):
      if i != j:
        if a[i][j] > largestElement:
          largestElement = a[i][j]
          k = i; l = j
          v = [k, l, largestElement]
   
  return v
def NormMatrix(a, rows, cols):
  norm = 0
  for i in range(rows):
    for j in range(cols):
      if(i != j):
        norm += a[i][j] * a[i][j]
  
  return math.sqrt(norm)


def rotation(k, l, a):
    c = 0; s = 0;tau = 0; t = 0
    if a[k][k] == a[l][l]:
      c = math.cos(math.pi/4)
      s = math.sin(math.pi/4)
    else:
      tau = (a[k][k]-a[l][l])/(2*a[k][l])
      if tau > 0:
        t = 1/(abs(tau) + math.sqrt(1+tau*tau))
      else:
        t = -1/(abs(tau) + math.sqrt(1+tau*tau))
      c = 1/math.sqrt(1+t*t)
      s = c*t

      tmp_jk = a[k][l]
      tmp_jj = a[k][k]

      a[k][l] = (c*c - s*s)*tmp_jk + s*c(a[l][l] - a[k][k])
      a[l][k] = a[k][l]
      a[k][k] = c*c*tmp_jj + 2*s*c*tmp_jk + s*s*a[l][l]
      a[l][l] = s*s*tmp_jj - 2*s*c*tmp_jk + c*c*a[l][l]

      tmp_jl = 0
      for i in range(rows):
        if i != k and i != l:
          tmp_jl = a[k][i]
          a[k][i] = c*tmp_jl + s*a[l][i]
          a[l][i] = s*tmp_jl - c*a[k][i]
          a[i][k] = a[k][i]
          a[i][l] = a[l][i]

def Jacobirotation(a, rows, cols, eps):
  k = 0; l = 0
  largestvalue = 0
  x = [k,l,largestvalue]
  x = largestElement(a, k, l)
  
  max = x[2]
  norm = NormMatrix(a, rows, cols)
  tol = eps*norm

  while NormMatrix(a, rows, cols) > tol:
    rotation(x[0], x[1], a)
    max = largestElement(a, x[0], x[1])



a = np.array([[1, 5, 7], 
      [5, 3, -5],
      [7, -5, 1]])

rows = len(a);  
cols = len(a[0]); 

egienvalues = 0
egienvectors = 0

if rows == cols:
  # make a transpose matrix of the given square matrix 
  trans_M = np.empty((rows, cols))
  for i in range(rows):
    for j in range(cols):
      trans_M[i][j] = a[j][i]

  # check whether a given matrix is symmetric or not
  flag = 0
  for i in range(rows):
    for j in range(cols):
      if a[i][j] != trans_M[i][j]:
        flag = 1
        break
  
  if flag ==  0:
    # given matrix is symmetric and now find it's egien values and vector using Jacobi Power Iteration 
    
    #step 1
    #Find the largest non-diagonal matrix
    
    #k = 0; l = 0
    #x = [k,l]
    #x = largestElement(a, k, l)
    
    #k = x[0]
    #l = x[1]
    
    #print(largestElement)

    #step 2
    # Find the rotational angle .
    
    #roatation(k, l, a)
    Jacobirotation(a, rows, cols, 1e-5)


  else:
    egienvalues,egienvectors = eig(a)

else:
  egienvalues,egienvectors = eig(a)
