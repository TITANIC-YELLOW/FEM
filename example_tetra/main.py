import HCAC
import read_tetra_dat
from multiprocessing import freeze_support

points, cells, boundaries = read_tetra_dat.getdata('D:/pythonProject3/example_tetra/Tetra.dat')

if __name__ == '__main__':
    freeze_support()

    x = HCAC.Solver(points, cells, boundaries)

    x.setting_element_type('tetrahedron')
    x.setting_time_step(10)
    x.setting_total_time(1000)
    x.setting_density(7900)
    x.setting_specific_heat(489)
    x.setting_conductivity([37,37,37])
    x.setting_heat_transfer_coefficient(100)
    x.setting_environment_temperature(1000)
    x.setting_initial_temperature(20)

    ans = x.solve()

