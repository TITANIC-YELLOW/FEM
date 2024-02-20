import numpy as np

num_cells = 23469
num_points = 4973

def get_points_data(filename:str) -> None:
    fp = open(filename)
    lines = fp.readlines()
    fp.close()

    coordinates = False

    Nodes_New = np.zeros(num_points * 3, dtype=np.float64).reshape(num_points, 3)

    for data in lines:
        data_list = [data[0:10], data[10:30], data[30:50], data[50:]]
        if data_list:
            if data_list[0] == "connectivi":
                connectivity = True
                continue
            if data_list[0] == "coordinate":
                coordinates = True
                connectivity = False
                continue

            if coordinates:
                try:
                    index = int(data_list[0]) - 1
                    Node_coo_list = []
                    for Node_str in data_list[1:]:
                        if len(Node_str[0:17].strip()) < 16:
                            Node_coo_list = [1., 1., 1.]
                            continue
                        else:
                            exponent_location = Node_str[1:].find("+")
                            if exponent_location == -1:
                                exponent_location = Node_str[1:].find("-")
                            exponent_location += 1
                            Node_coo = Node_str[0:exponent_location] + "e" + Node_str[exponent_location:]
                            Node_coo = float(Node_coo)
                            Node_coo_list.append(Node_coo)
                    Nodes_New[index, :] = Node_coo_list
                except ValueError:
                    coordinates = False
                    pass


    np.save('points.npy', Nodes_New)

# get_points_data('test_job1.dat')

n = 20860-24+1
def get_cells1_data(filename:str) -> None:
    fp = open(filename)
    lines = fp.readlines()
    fp.close()

    connectivity = []
    for i in range(n):
        data = lines[i + 23]

        data2 = data.strip('\n')
        data3 = data2.split()
        for s in data3:
            if s.isdigit():
                connectivity.append(int(s))
            elif s.isdigit() is False and s != '':
                print(s)


    cells = np.array(connectivity).reshape(-1, 6)
    cells = np.delete(cells, [0, 1], axis=1)
    np.save('cells1.npy', cells)

def get_cells2_data(filename:str) -> None:
    fp = open(filename)
    lines = fp.readlines()
    fp.close()

    connectivity = []
    for i in range(num_cells-n):
        data = lines[i + 20862]

        data2 = data.strip('\n')
        data3 = data2.split()
        for s in data3:
            if s.isdigit():
                connectivity.append(int(s))
            elif s.isdigit() is False and s != '':
                print(s)


    cells = np.array(connectivity).reshape(-1, 6)
    cells = np.delete(cells, [0, 1], axis=1)
    np.save('cells2.npy', cells)


def get_boundry_data(filename:str):
    fp = open(filename)
    lines = fp.readlines()
    fp.close()

    boundary = []
    for i in range(44):
        data = lines[i]

        data2 = data.strip('\n')
        data3 = data2.split()
        for s in data3:
            if s.isdigit():
                boundary.append(int(s))
            elif s.isdigit() is False and s != '':
                print(s)

    b = np.array(boundary)
    np.save('boundary.npy', b)

# get_cells1_data('test_job1.dat')
# get_cells2_data('test_job1.dat')
get_boundry_data('boundry.txt')