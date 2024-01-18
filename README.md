# FiniteElementMethod
This repository contains codes used in FEM.  
Python is used as the primary language.  
## Integral in Pyramid
Taking heat conduction as an example, if we want to calculate the stiffness matrix, mass matrix, and load vector of a pyramid shaped element, we need to perform triple integration of the correlation function f (x, y, z) within the element.  


At this point, the **PyramidInt64.pyd** file came in handy. In this file, a function IntInPyramid (func) was defined, which takes a function name func as a parameter and returns a float type result, which is the integration value of the function in the standard integral domain.  

### the standard integral domain of pyramid shaped element
```python
(-1,-1, 0)
( 1,-1, 0)
( 1, 1, 0)
(-1, 1, 0)
( 0, 0, 1)
``` 

### Notes
* If you want to convert any pyramid type integral domain into a standard integral domain, you need to perform coordinate transformation on the original function, and IntInPyramid (func) can only handle functions in the standard integral domain.  
* Make sure that you have installed numpy.  

* PyramidInt64.pyd can only be operated in Python3.9.  

### Examples
code1:  
```python  
import PyramidInt64

def func(x,y,z):
    return 1/(x+2)

print(PyramidInt64.IntInPyramid(func))
```
result：  
`0.704165948193644`  

code2:  
```python
import PyramidInt64

def func(x,y,z):
    return 1

print(PyramidInt64.IntInPyramid(func))
```
result：  
`1.3333333333333357`  
It is obvious that  code 2 calculates the volume of the standard pyramid, which is' 4/3 '.  

### integral precision
For polynomials, the **IntInPyramid** function can handle the highest order of`x^7*y^7*z^7`.  
### execution speed
Compare **IntInPyramid** with scipy.integrate.tplquad,let's see the difference between them:  

code3:  
```python
import time
from scipy import integrate
import PyramidInt64
def func(x,y,z):
    return x**2 + y**2 + z**2

t1 = time.time()
print(PyramidInt64.IntInPyramid(func))
t2 = time.time()
print(t2-t1)

t1 = time.time()
print(integrate.tplquad(func,-1,1,-1,1,
                        0,lambda x,y:1.0 - (abs(x) + abs(y) + abs(abs(x) - abs(y))) / 2)[0])
t2 = time.time()
print(t2-t1)
```
result:  
```python
0.6666666666666666
0.0
0.6666667796697336
6.591209888458252
```
