import numpy as np


'''V_tetra returns the volume of a tetrahedron.'''
def V_tetra(coor:np.ndarray) -> float:...


'''m_tetra returns the mass matrix of a tetrahedron.'''
def m_tetra(coor:np.ndarray, p:float, c:float) -> np.ndarray:...


'''k_tetra returns the stiffness matrix of a tetrahedron.'''
def k_tetra(coor:np.ndarray, ThermalConductivity:list) -> np.ndarray:...


'''edge_tri is used to correct the stiffness matrix affected by the third type of boundary conditions.'''
def edge_tri(coor:np.ndarray, h:float) -> np.ndarray:...


'''f3_tri returns a triangular element's load vector related to the third type of boundary condition.'''
def f3_tri(coor:np.ndarray, h:float, Tf:float) -> np.ndarray:...

'''fQ_tetra returns a tetrahedral element's load vector related to the internal heat source.'''
def fQ_tetra(c:float, Q:float, coor:np.ndarray) -> np.ndarray:...


'''V_prism returns the volume of a prism.'''
def V_prism(coor:np.ndarray) -> float:...


'''V_prism27 returns the volume of a prism, which is more accurate than the result of V_prism.'''
def V_prism27(coor:np.ndarray) -> float:...


'''k_prism returns the stiffness matrix of a prism.'''
def k_prism(coor:np.ndarray, ThermalConductivity:list) -> np.ndarray:...


'''k_prism27 returns the stiffness matrix of a prism, which is more accurate than the result of k_prism.'''
def k_prism27(coor:np.ndarray, ThermalConductivity:list) -> np.ndarray:...


'''m_prism returns the mass matrix of a prism.'''
def m_prism(coor:np.ndarray, p:float, c:float) -> np.ndarray:...


'''m_prism27 returns the mass matrix of a prism which is more accurate than the result of m_prism.'''
def m_prism27(coor:np.ndarray, p:float, c:float) -> float:...


'''fQ_prism returns a prism element's load vector related to the internal heat source.'''
def fQ_prism(c:float, Q:float, coor:np.ndarray) -> np.ndarray:...


'''V_pyramid returns the volume of a pyramid.'''
def V_pyramid(coor:np.ndarray):...


'''k_pyramid returns the stiffness matrix of a pyramid.'''
def k_pyramid(coor:np.ndarray, ThermalConductivity:list) -> np.ndarray:...


'''m_pyramid returns the mass matrix of a pyramid.'''
def m_pyramid(coor:np.ndarray, p:float, c:float) -> np.ndarray:...


'''fQ_pyramid returns a pyramid element's load vector related to the internal heat source.'''
def fQ_pyramid(c:float, Q:float, coor:np.ndarray) -> np.ndarray:...


'''V_hexa returns the volume of a hexahedron.'''
def V_hexa(coordinate:np.ndarray) -> float:...


'''m_hexa returns the mass matrix of a hexahedron.'''
def m_hexa(coordinate:np.ndarray, p:float, c:float) -> np.ndarray:...


'''k_hexa returns the stiffness matrix of a hexahedron.'''
def k_hexa(coordinate:np.ndarray, ThermalConductivity:list) -> np.ndarray:...


'''fQ_hexa returns a hexahedron element's load vector related to the internal heat source.'''
def fQ_hexa(c:float, Q:float, coor:np.ndarray) -> np.ndarray:...


'''f3_quad returns a quadrilateral element's load vector related to the third type of boundary condition.'''
def f3_quad(h:float, Tf:float, coor:np.ndarray) -> np.ndarray:...


'''edge_quad is used to correct the stiffness matrix affected by the third type of boundary conditions.'''
def edge_quad(h:float, coor:np.ndarray) -> np.ndarray:...

