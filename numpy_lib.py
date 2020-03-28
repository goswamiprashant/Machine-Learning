import numpy as np
l=[1,2,3,4,5,6]
print(type(l))
x=np.array(l)#  toconvert in numpy array
print(type(x))
y=np.arange(0,20,2)     # works as range fun
print(y)
print(np.zeros(5))
print(np.zeros((2,5)))# for creation pf m*n zero matrix consist float datatype
print(np.ones(5))
print(np.ones((2,5)))# for creation pf m*n Unity matrix consist float datatype
print(np.random.randint(1,7))# for generating random num ,array,matrix
print(np.random.randint(1,7,2))
print(np.random.randint(1,100,(10,10)))
z=np.arange(0,100).reshape(10,10)  # ffor reshaping a array to matrix or size of matrix
print(z)
print(z.max(),z.argmax())  # to get max value and its index
print(z.min(),z.argmin())# to get min value and its index
print(z.mean())# to get the mean
# accessing of elements
print(z[0,3])
print(z[0,:],z[0])# row
print(z[:,0])     #column
#print(np.arange(0,100).reshape(10,10))
#slicing of array
print(z[0:4,0:4])