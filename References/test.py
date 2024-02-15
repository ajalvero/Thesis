import numpy as np
#from scipy import stats

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]                    # set the middle as pivot
    left = [x for x in arr if x < pivot]          # find all x smaller than pivot in arr
    middle = [x for x in arr if x == pivot]       # find all x equal to pivot in arr
    right = [x for x in arr if x > pivot]         # find all x larger than pivot in arr
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3, 6, 8, 10, 1, 2, 1]))

# numbers
x = 3  # x is a variable which is assigned with a numeric value: 3
y = 1.0

print(x, type(x))  #print() is a built-in function for printing 
print(y, type(y))

print(x + 1)  # Addition
print(x * 2)  # Multiplication
print(x ** 2) # Exponentiation
print(x // 2) # Floor division

print(y)
y += 1  # Same as y = y + 1
print(y)
y *= 2  # Same as y = y * 2
print(y)

#bool

t, f = True, False  # Python can do multiple assignments in one line
print(type(t), type(f))

print(t and f) # Logical AND;
print(t or f)  # Logical OR;
print(not t)   # Logical NOT;

x, y, z = 3, 1.0, 3.0
print(x < y)   # Return True if x is LESS than y
print(x == z)  # Return True if x is EQUAL to y

#strings
h = 'hello'       # String literals can use single quotes
w = "world"       # or double quotes
print(h, len(h))  # Built-in function len() return the length of elements

hw = h + ' ' + w  # String concatenation
print(hw)

hw1 = '{} {}! {}'.format(h, w, 2023)             # String formatting by sequence
print(hw1)
hw2 = '{1} {0}! {2} {2:.2f}'.format(h, w, 2023)  # String formatting by specifying orders and formats
print(hw2)

print(h)
print(h.upper())       # Convert a string to uppercase; prints "HELLO"
print(h.replace('l', '(ell)'))  # Replace all instances of one substring with another

#containers

ls = [3, 1, 'foo']  # This list contains three elements with different types
print(ls, len(ls))

ls.append('bar') # Add a new element to the end of the list
print(ls)
ls.pop()         # Remove and return the last element of the list
print(ls)

print(ls[2])     # Indexing 3rd element; list indexing starts from 0
print(ls[-1])    # Negative indices count from the end of the list

nums = [0, 1, 2, 3, 4, 5, 6]
print(nums)
print(nums[2:4])    # Get a slice from index 2 to 4 (exclusive)
print(nums[2:])     # Get a slice from index 2 to the end
print(nums[:-1])    # Slice indices can also be negative
nums[2:4] = [8, 9]  # Assign a new sublist to a slice
print(nums)

print(nums[:-1:2])  # Get a slice from index 0 to -1 (exclusive) in a step length of 2
print(nums[::-1])   # Get a slice of whole list in reverse order

#dictionaries

d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print(d['cat'])       # Get an value from a dictionary

print('fish' in d)  # `in` is the membership operator to check the presence
d['fish'] = 'wet'   # Set a new entry in a dictionary
print('fish' in d)

print(d.get('monkey', 'N/A'))  # Get a value with a default
print(d.get('fish', 'N/A'))    # Get a value with a default

#tuples

t1 = (5, 6)  # Create a tuple

print(t1, type(t1))


#control flow

#conditions
x, y = 10, 12

if x > y:
    print("x>y")  # Four blanks before the algorithm
elif x < y:
    print("x<y")  # Four blanks before the algorithm
else:
    print("x=y")  # Four blanks before the algorithm

#loops

list_of_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

for list1 in list_of_lists: # Iterate over elements in list_of_lists
    print(list1)            # Four blanks before the algorithm
print('Bye')                # Without four blanks, this is not a part of iterations


for i in range(0, 2):
    print(list_of_lists[i])

d = {'person': 2, 'cat': 4, 'spider': 8}

for animal, legs in d.items():
    print('A {} has {} legs'.format(animal, legs))

i = 1
while i < 3:    # Iterate when i smaller than 3
    print(i**2) # Four blanks before each line of algorithm
    i += 1      # Four blanks before

#list comprehension and dictionary comprehension

nums = [0, 1, 2, 3, 4]
squares = [x**2 for x in nums]
print(squares)

even_squares = [x**2 for x in nums if x % 2 == 0]
print(even_squares)

even_num_to_square = {x: x**2 for x in nums if x % 2 == 0}
print(even_num_to_square)

#functions

def sign(x):   # Define a function with one argument x
    '''determine the sign of a single value'''
    if x > 0:  # four blanks before each line within the function body
        return 'positive'  # another four blanks within `if` expressions
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]: 
    print("{} is {}".format(x, sign(x)))  # Use parenthesis to run functions 

def hello(name, loud=False):
    if loud:
        print('HELLO, {}'.format(name.upper()))
    else:
        print('Hello, {}!'.format(name))

hello('Bob')  # Without specifin second argument, function would take default for it
hello('Fred', loud=True)

y = lambda x: x**3 + x**2 + x  # define a simple lambda funtion
y(-1)                          # call it

#classes and objects

class Car():
    
    def __init__(self, company, model, year):  
        """initialize the properties of a car"""
        self.company = company  # claim one property of object
        self.model = model
        self.year = year
        self.odometer = 0
        
    def get_info(self):  # functions are the methods of object
        """return car information in a string"""
        car_info = "{} {} {}".format(self.year, self.company, self.model)
        return car_info
        
    def run(self, distance):
        """run car for a distance"""
        self.odometer += distance
    
    def read_odometer(self):
        """return the distances the car has run through"""
        odo_info = "This car has run {} km.".format(self.odometer)
        return odo_info
    
my_lovely_car = Car("Tesla", "Model 3", 2022)  # calling class name would trigger __init__ to create an instance
print(my_lovely_car.company)     # retrive a property
print(my_lovely_car.get_info())  # call a method

print(my_lovely_car.read_odometer())
my_lovely_car.run(5.5)
print(my_lovely_car.read_odometer())

Car.run(my_lovely_car, 3)  # methods can also be called under the class name
print(Car.read_odometer(my_lovely_car))

