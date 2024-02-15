import numpy as np
print(np.__version__)


#Array and its Creation

a = np.array([1, 2, 3])  # Create a rank 1 array from a list
print(a, type(a))
print(a.shape, a.dtype, a[0])

b = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # Create a rank 2 array
print(b, b.shape, b[1, 2])

c = np.array([[[111, 112, 113, 114], [121, 122, 123, 124]],
              [[211, 212, 213, 214], [221, 222, 223, 224]],
              [[311, 312, 313, 314], [321, 322, 323, 324]]])
                                            # Create a rank 3 array
print(c, c.shape, c[0, 1, 2])

d = np.arange(5, 50, 10)  # Create an array starting at 5, ending at 50, with a step of 10
d = np.zeros((2, 2))      # Create an array of all zeros with shape (2, 2)
d = np.ones((1, 2))       # Create an array of all ones with shape (1, 2)
d = np.random.random((3, 1))  # Create an array of random values with shape (3, 1)
# Try printing them
print(d)

#Array Indexing

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a[:2, 1:3])  # Slice 1st to 2nd rows and 2nd to 3rd columns
print(a[:, ::2])   # Slice all odd columns

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
b1 = a[:2, 1:3]
b1[0, 0] = 77    # b[0, 0] will be the same piece of data as a[0, 1]
print(b1[0, 0], a[0, 1])

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
b2 = a[:2, 1:3].copy()
b2[0, 0] = 77
print(b2[0, 0], a[0, 1])

#integer array indexing
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(a[[0, 1, 2], [0, 1, 0]]) # Integer indexing

# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))

row = [0, 1, 2]  # Explicitly express row indices
col = [0, 1, 0]  # and col indices
a[row, col] += 1000  # Only operate on specific elements
print(a)

#boolean array indexing
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
bool_idx = (a > 8)  # Find the elements of a that are bigger than 2;
                    # this returns a numpy array of Booleans of the same
                    # shape as a, where each slot of bool_idx tells
                    # whether that element of a is > 2.

print(bool_idx)
print(a[bool_idx])  # Boolean array indexing, return rank 1 array for True positions
#or
a[a > 8]

#Array Manipulation

a = np.arange(12)
print(a)
print(a.reshape((3, 4)))
print(np.reshape(a, (3, 4)))  # use the class method and put object as 1st argument is the same

a = np.arange(12).reshape((3, 4))
print("transpose through property\n", a.T)          # property is like a variable
print("transpose through method\n", a.transpose())  # method is like a function

a = np.arange(12).reshape((3, 4))
b = np.arange(8).reshape((2, 4))
c = np.arange(6).reshape((3, 2))

ac = np.hstack((a, c))
ab = np.vstack((a, b))
print(ac)
print(ab)

#Array Math

x = np.array([[1, 2], [3, 4]], dtype=np.float64) # Set data types of elements by dtype
y = np.array([[5, 6], [7, 8]], dtype=np.float64)

# Elementwise sum; both produce an array
print(x + y)
print(np.add(x, y))

# Elementwise square root; produces an array
print(np.sqrt(x))
# Elementwise natural logarithm; produces an array
print(np.log(x))

x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

# Inner product of vectors
print(x[0, :].dot(y[0, :]))
print(np.dot(x[0, :], y[0, :]))
# Matrix / matrix product
print(x.dot(y))
print(np.dot(x, y))

#Aggregation Calculations

x = np.array([[1, 2, 3], [4, 5, 6]])

print(np.sum(x))          # Sum of all elements; produce a value
print(np.sum(x, axis=0))  # Sum along axis 0 (column); produce a lower rank array
print(x.sum(axis=1))      # Sum along axis 1 (row); produce a lower rank array
print(x.mean(axis=1))
# Try others!

d1 = np.arange(1, 5)
d2 = np.arange(1, 13).reshape((3, 4))
d3 = np.arange(1, 25).reshape((2, 3, 4))

print("Minimum along axis 0:")
print(d3.min(axis=0))  # ❓: Why we have this result?

#Broadcasting

x = np.arange(1, 11).reshape((2, 5))
x_norm = x / 10
print(x_norm)

sign = np.array([-1, 1]).reshape((2, 1))
x_signed = x * sign
print(x_signed)

A = np.arange(1, 6)
B = np.arange(1, 3).reshape((2, 1))  # ❓: why we need reshape?

Result = A * B
print(Result)

data = np.array([[25.1, 27.5, 32.3, 24.4, 28.2, 29.8, 30.1],  # a row of temperature
                 [20.0,  2.5,  8.5, 10.0,  0.5, 28.4, 12.5]]) # a row of precipitation
norm = (data - np.min(data, axis=1).reshape(2, 1)) / np.ptp(data, axis=1).reshape(2, 1)
print(norm)


# Your solution goes here.