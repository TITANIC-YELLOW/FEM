import solve
import readDAT
from multiprocessing import freeze_support

points, cells, boundaries = readDAT.getdata('model1_job1.dat')

if __name__ == '__main__':
    freeze_support()

    x = solve.Solve(points, cells, boundaries)

    x.setting_element_type('hexahedron')
    x.setting_time_step(1)
    x.setting_total_time(100)
    x.density = 7850
    x.specific_heat = 460
    x.conductivity = [45, 45, 45]
    x.heat_transfer_coefficient = 2000
    x.environment_temperature = 100
    x.initial_temperature = 1000

    ans = x.solve()
