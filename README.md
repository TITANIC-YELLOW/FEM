# FiniteElementMethod
This repository contains codes used in FEM.  
这个仓库包含了FEM中使用到的代码。  
Python is used as the primary language.  
使用了Python作为主要的语言。  
## Integral in Pyramid（金字塔型单元内的积分）
Taking heat conduction as an example, if we want to calculate the stiffness matrix, mass matrix, and load vector of a pyramid shaped element, we need to perform triple integration of the correlation function f (x, y, z) within the element.  
以热传导为例，如果要计算金字塔形单元的刚度矩阵、质量矩阵和载荷向量，我们需要对相关函数f(x,y,z)在单元内进行三重积分。  

At this point, the **PyramidInt64.pyd** file came in handy. In this file, a function IntInPyramid (func) was defined, which takes a function name func as a parameter and returns a float type result, which is the integration value of the function in the standard integral domain.  
在这个时候**PyramidInt64.pyd**文件就派上了用场，在这个文件中定义了一个函数IntInPyramid(func)，它获取一个函数名func作为参数，返回一个float类型的结果，也就是该函数在标准积分域里的积分值。  

### the standard integral domain of pyramid shaped element（金字塔型单元的标准积分域）
`(-1,-1,0),
(1,-1,0),
(1,1,0),
(-1,1,0),
(0,0,1)`  

### Notes（注意事项）
* If you want to convert any pyramid type integral domain into a standard integral domain, you need to perform coordinate transformation on the original function, and IntInPyramid (func) can only handle functions in the standard integral domain.  
如果要将任意的金字塔型积分域转换为标准积分域，需要对原函数进行坐标变换，IntInPyramid(func)只能处理标准积分域里的函数。
* Make sure that you have installed numpy.  
确保你已经安装了numpy。
* PyramidInt64.pyd can only be operated in Python3.9.  
PyramidInt64.pyd只能在Python3.9中运行。
### Examples（示例代码）
示例代码1：  
```python  
import PyramidInt64

def func(x,y,z):
    return 1/(x+2)

print(PyramidInt64.IntInPyramid(func))
```
结果为：  
`0.704165948193644`  

示例代码2：  
```python
import PyramidInt64

def func(x,y,z):
    return 1

print(PyramidInt64.IntInPyramid(func))
```
结果为：  
`1.3333333333333357`  
很显然示例代码2所求的就是标准金字塔的体积，即：`4/3`。  

### 积分精度
对于多项式，IntInPyramid函数可以处理最高阶数为`x^7*y^7*z^7`。  

