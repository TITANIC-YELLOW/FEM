import numpy as np


def getdata(filename: str) -> tuple:
    fp = open(filename)
    lines = fp.readlines()
    fp.close()

    num_elements = 0
    num_points = 0

    row_connectivity = 0
    row_coordinates = 0

    for i in range(len(lines)):
        if lines[i][0:6] == 'sizing':
            # remove '\n'
            data = lines[i].strip()  # type:str
            # remove 'space'
            list_data = data.split()  # type:list

            num_elements = int(list_data[2])
            num_points = int(list_data[3])

        elif lines[i][0:12] == 'connectivity':
            row_connectivity = i

        elif lines[i][0:11] == 'coordinates':
            row_coordinates = i
            break


    # get connectivity data
    cell = []
    for j in range(row_connectivity+2, row_connectivity+num_elements+2):
        data2 = lines[j].strip()
        list_data2 = data2.split()

        for s in list_data2:
            cell.append(int(s))

    cell2 = np.array(cell).reshape(num_elements, -1)
    cell2 = np.delete(cell2, [0, 1], axis=1)


    # get coordinates data
    points = []
    for k in range(row_coordinates+2, row_coordinates+num_points+2):
        data3 = lines[k].strip()
        list_data3 = [data3[-60:-40], data3[-40:-20], data3[-20:]]

        for key in range(3):
            try:
                points.append(float(list_data3[key][0:-2] + 'e' + list_data3[key][-2:]))
            except:
                print(list_data3[key][0:-2] + 'e' + list_data3[key][-2:])
                points.append(0.0)

    coordinate = np.array(points).reshape(num_points, 3)

    return cell2, coordinate


# ans,ans2 = getdata('axis_quad.dat')
# # ans,ans2 = getdata('model1_job1.dat')
#
# np.savetxt('ccccc.txt',ans2)
# print(ans.shape[0], ans.shape[1])
# print(ans2.shape[0], ans2.shape[1])