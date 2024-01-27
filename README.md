# FiniteElementMethod
This repository contains codes used in FEM.

*Python* is used as the primary language.
## HCMVG.cp39-win_amd64.pyd
*HCMVG* was created to generate 3D-elements(tetrahedron,hexahedron,prism,pyramid)'s matrices and vectors in *heat conduction*.  
### Notes
* *HCMVG.cp39-win_amd64.pyd* may contain mistakes.
* Make sure that you have installed *numpy*.  
* *HCMVG.cp39-win_amd64.pyd* can **only** be operated in *Python3.9*.

### Function Explanation
#### Tetrahedron
* *V_tetra(coor:np.ndarray) -> float:*

In this function, *coor* refers to the coordinates of the four vertices of a tetrahedron and the output is the volume of the tetrahedron element.

Example:
```python
import HCMVG
import numpy as np

coor = np.array([[0, 0, 0],
                 [1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1]])
ans = HCMVG.V_tetra(coor)
print(ans)
```
result: `0.16666666666666666`  

* *m_tetra(coor:np.ndarray, p:float, c:float) -> np.ndarray:*

In this function, *coor* refers to the coordinates of the four vertices of a tetrahedron, *p* is the density of the material, *c* is the specific heat of the material and the output is the mass matrix of the tetrahedron element.

Example:  
```python
import HCMVG
import numpy as np

coor = np.array([[0, 0, 0],
                 [1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1]])
ans = HCMVG.m_tetra(coor,1,1)
print(ans)
```
result: 
```python
[[0.01666667 0.00833333 0.00833333 0.00833333]
 [0.00833333 0.01666667 0.00833333 0.00833333]
 [0.00833333 0.00833333 0.01666667 0.00833333]
 [0.00833333 0.00833333 0.00833333 0.01666667]]
```

* *k_tetra(coor:np.ndarray, ThermalConductivity:list) -> np.ndarray:*

In this function, *coor* refers to the coordinates of the four vertices of a tetrahedron, *ThermalConductivity* is the thermal conductivity of the material(for isotropic materials, *ThermalConductivity* equals a constant multiplies the *list[1, 1, 1]*), and the output is the stiffness matrix of the tetrahedron element.

Example:  
```python
import HCMVG
import numpy as np

coor = np.array([[0, 0, 0],
                 [1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1]])
ThermalConductivity = [1, 1, 1]
ans = HCMVG.k_tetra(coor, ThermalConductivity)
print(ans)
```
result:
```python
[[ 0.5        -0.16666667  0.         -0.33333333]
 [-0.16666667  0.16666667  0.          0.        ]
 [ 0.          0.          0.33333333 -0.33333333]
 [-0.33333333  0.         -0.33333333  0.66666667]]
```

#### Hexahedron
* *V_hexa(coor:np.ndarray) -> float:*

In this function, *coor* refers to the coordinates of the eight vertices of a hexahedron and the output is the volume of the hexahedron element.

Example:
```python
import HCMVG
import numpy as np

coor = np.array([[0, 0, 0],
                 [1, 0, 0],
                 [1, 2, 0],
                 [0, 2, 0],
                 [0, 0, 3],
                 [1, 0, 3],
                 [1, 2, 3],
                 [0, 2, 3]])
ans = HCMVG.V_hexa(coor)
print(ans)
```
result:
```python
6.0
```

* m_hexa

In this function, *coor* refers to the coordinates of the eight vertices of a hexahedron, *p* is the density of the material, *c* is the specific heat of the material and the output is the mass matrix of the hexahedron element.

Example:
```python
import HCMVG
import numpy as np

coor = np.array([[0, 0, 0],
                 [1, 0, 0],
                 [1, 2, 0],
                 [0, 2, 0],
                 [0, 0, 3],
                 [1, 0, 3],
                 [1, 2, 3],
                 [0, 2, 3]])
ans = HCMVG.m_hexa(coor, 1, 1)
print(ans)
```
result:
```python
[[0.22222222 0.11111111 0.05555556 0.11111111 0.11111111 0.05555556
  0.02777778 0.05555556]
 [0.11111111 0.22222222 0.11111111 0.05555556 0.05555556 0.11111111
  0.05555556 0.02777778]
 [0.05555556 0.11111111 0.22222222 0.11111111 0.02777778 0.05555556
  0.11111111 0.05555556]
 [0.11111111 0.05555556 0.11111111 0.22222222 0.05555556 0.02777778
  0.05555556 0.11111111]
 [0.11111111 0.05555556 0.02777778 0.05555556 0.22222222 0.11111111
  0.05555556 0.11111111]
 [0.05555556 0.11111111 0.05555556 0.02777778 0.11111111 0.22222222
  0.11111111 0.05555556]
 [0.02777778 0.05555556 0.11111111 0.05555556 0.05555556 0.11111111
  0.22222222 0.11111111]
 [0.05555556 0.02777778 0.05555556 0.11111111 0.11111111 0.05555556
  0.11111111 0.22222222]]
```

* k_hexa

In this function, *coor* refers to the coordinates of the eight vertices of a hexahedron, *ThermalConductivity* is the thermal conductivity of the material(for isotropic materials, *ThermalConductivity* equals a constant multiplies the *list[1, 1, 1]*), and the output is the stiffness matrix of the hexahedron element.

Example:
```python
import HCMVG
import numpy as np

coor = np.array([[0, 0, 0],
                 [1, 0, 0],
                 [1, 2, 0],
                 [0, 2, 0],
                 [0, 0, 3],
                 [1, 0, 3],
                 [1, 2, 3],
                 [0, 2, 3]])
ThermalConductivity = [1, 1, 1]
ans = HCMVG.k_hexa(coor, ThermalConductivity)
print(ans)
```
result:
```python
[[ 0.90740741 -0.5462963  -0.39814815  0.2037037   0.34259259 -0.3287037
  -0.22685185  0.0462963 ]
 [-0.5462963   0.90740741  0.2037037  -0.39814815 -0.3287037   0.34259259
   0.0462963  -0.22685185]
 [-0.39814815  0.2037037   0.90740741 -0.5462963  -0.22685185  0.0462963
   0.34259259 -0.3287037 ]
 [ 0.2037037  -0.39814815 -0.5462963   0.90740741  0.0462963  -0.22685185
  -0.3287037   0.34259259]
 [ 0.34259259 -0.3287037  -0.22685185  0.0462963   0.90740741 -0.5462963
  -0.39814815  0.2037037 ]
 [-0.3287037   0.34259259  0.0462963  -0.22685185 -0.5462963   0.90740741
   0.2037037  -0.39814815]
 [-0.22685185  0.0462963   0.34259259 -0.3287037  -0.39814815  0.2037037
   0.90740741 -0.5462963 ]
 [ 0.0462963  -0.22685185 -0.3287037   0.34259259  0.2037037  -0.39814815
  -0.5462963   0.90740741]]
```

#### Prism
* V_prism
Example:
```python
```
result:
```python
```
* m_prism
Example:
```python
```
result:
```python
```
* k_prism
Example:
```python
```
result:
```python
```
#### Pyramid
* V_pyramid
Example:
```python
```
result:
```python
```
* m_pyramid
Example:
```python
```
result:
```python
```
* k_pyramid
Example:
```python
```
result:
```python
```
## PyramidInt64.pyd
A function named *IntInPyramid* in *PyramidInt64.pyd* was created to calculate triple integration of the correlation function f (x, y, z) within the pyramid-shaped element.

It takes a function name as a parameter,

and returns a *float* type result, which is the integration value of the function in the standard integral domain.  

### The standard integral domain of pyramid-shaped element
```python
(-1,-1, 0)
( 1,-1, 0)
( 1, 1, 0)
(-1, 1, 0)
( 0, 0, 1)
``` 

### Notes
* *IntInPyramid* can only handle functions in the standard integral domain, so you need to perform coordinate transformation on the original function if you the integral domain is not standard.
* Make sure that you have installed *numpy*.  
* *PyramidInt64.pyd* can only be operated in *Python3.9*.  

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
It is obvious that  code 2 calculates the volume of the standard pyramid, which is `4/3`.  

### Integral Precision
For polynomials, the *IntInPyramid* function can handle the highest order of`x^7*y^7*z^7`.  
### Execution Speed
Compare *IntInPyramid* with *scipy.integrate.tplquad*:  

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
`
0.6666666666666666  
0.0  
0.6666667796697336  
6.591209888458252  
`
