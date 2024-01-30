import numpy as np


'''V_tetra returns the volume of a tetrahedron.'''
def V_tetra(CoordinateArray:np.ndarray) -> float:...


'''m_tetra returns the mass matrix of a tetrahedron.'''
def m_tetra(CoordinateArray:np.ndarray, p:float, c:float) -> np.ndarray:...


'''k_tetra returns the stiffness matrix of a tetrahedron.'''
def k_tetra(CoordinateArray:np.ndarray, ThermalConductivity:list) -> np.ndarray:...


'''edge_tri is used to correct the stiffness matrix affected by the third type of boundary conditions.'''
def edge_tri(CoordinateArray:np.ndarray, h:float) -> np.ndarray:...


'''f3_tri returns a triangular element's load vector related to the third type of boundary condition.'''
def f3_tri(CoordinateArray:np.ndarray, h:float, Tf:float) -> np.ndarray:...

'''fQ_tetra returns a tetrahedral element's load vector related to the internal heat source.'''
def fQ_tetra(CoordinateArray:np.ndarray, c:float, Q:float) -> np.ndarray:...


'''V_prism returns the volume of a prism.'''
def V_prism(CoordinateArray:np.ndarray) -> float:...


'''V_prism27 returns the volume of a prism, which is more accurate than the result of V_prism.'''
def V_prism27(CoordinateArray:np.ndarray) -> float:...


'''k_prism returns the stiffness matrix of a prism.'''
def k_prism(CoordinateArray:np.ndarray, ThermalConductivity:list) -> np.ndarray:...


'''k_prism27 returns the stiffness matrix of a prism, which is more accurate than the result of k_prism.'''
def k_prism27(CoordinateArray:np.ndarray, ThermalConductivity:list) -> np.ndarray:...


'''m_prism returns the mass matrix of a prism.'''
def m_prism(CoordinateArray:np.ndarray, p:float, c:float) -> np.ndarray:...


'''m_prism27 returns the mass matrix of a prism which is more accurate than the result of m_prism.'''
def m_prism27(CoordinateArray:np.ndarray, p:float, c:float) -> float:...


'''fQ_prism returns a prism element's load vector related to the internal heat source.'''
def fQ_prism(CoordinateArray:np.ndarray, c:float, Q:float) -> np.ndarray:...


'''V_pyramid returns the volume of a pyramid.'''
def V_pyramid(CoordinateArray:np.ndarray) -> float:...


'''k_pyramid returns the stiffness matrix of a pyramid.'''
def k_pyramid(CoordinateArray:np.ndarray, ThermalConductivity:list) -> np.ndarray:...


'''m_pyramid returns the mass matrix of a pyramid.'''
def m_pyramid(CoordinateArray:np.ndarray, p:float, c:float) -> np.ndarray:...


'''fQ_pyramid returns a pyramid element's load vector related to the internal heat source.'''
def fQ_pyramid(CoordinateArray:np.ndarray, c:float, Q:float) -> np.ndarray:...


'''V_hexa returns the volume of a hexahedron.'''
def V_hexa(CoordinateArray:np.ndarray) -> float:...


'''m_hexa returns the mass matrix of a hexahedron.'''
def m_hexa(CoordinateArray:np.ndarray, p:float, c:float) -> np.ndarray:...


'''k_hexa returns the stiffness matrix of a hexahedron.'''
def k_hexa(CoordinateArray:np.ndarray, ThermalConductivity:list) -> np.ndarray:...


'''fQ_hexa returns a hexahedron element's load vector related to the internal heat source.'''
def fQ_hexa(CoordinateArray:np.ndarray, c:float, Q:float) -> np.ndarray:...


'''f3_quad returns a quadrilateral element's load vector related to the third type of boundary condition.'''
def f3_quad(CoordinateArray:np.ndarray, h:float, Tf:float) -> np.ndarray:...


'''edge_quad is used to correct the stiffness matrix affected by the third type of boundary conditions.'''
def edge_quad(CoordinateArray:np.ndarray, h:float) -> np.ndarray:...

