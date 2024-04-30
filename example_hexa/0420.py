import HCAC
import readDAT
from multiprocessing import freeze_support

points, cells, boundaries = readDAT.getdata('D:/pythonProject3/test/model1_job1.dat')

if __name__ == '__main__':
    freeze_support()

    x = HCAC.Solver(points, cells, boundaries)

    x.setting_element_type('hexahedron')
    x.setting_time_step(1)
    x.setting_total_time(100)
    x.setting_density(7850)
    x.setting_specific_heat(460)
    x.setting_conductivity([45,45,45])
    x.setting_heat_transfer_coefficient(2000)
    x.setting_environment_temperature(100)
    x.setting_initial_temperature(1000)

    ans = x.solve()
