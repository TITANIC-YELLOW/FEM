import numpy as np

canshu = np.array([[6,5,1,2],[3,7,6,2],[3,4,8,7],
                   [5,8,4,1],[1,4,3,2],[6,7,8,5]])

def getdata(filename:str):
    fp=open(filename)
    lines = fp.readlines()
    fp.close()

    connectivity = False
    Elements_num = 0
    coordinates = False
    Nodes_num = 0
    facemt = False
    face_num = 0

    cell = []
    point = []
    face = []


    f = []

    for data in lines:
        # print(data)

        # data_list = [data[0:10], data[10:30], data[30:50], data[50:]]

        # if data_list:
        if data[0:10] == "connectivi":
            connectivity = True
            continue
        if data[0:10] == "coordinate":
            coordinates = True
            connectivity = False
            continue
        if data[0:6] == 'define':
            coordinates = False

        if len(data) == 71 and data[0:6] == 'define':
            facemt = True
            coordinates = False
            continue
            # print(aaaaaaaa)
        if data =='define              element             set                 apply6_elements\n':
            facemt = False
            break

        if connectivity:
            Elements_num += 1
            if Elements_num > 1:
                data = data.strip('\n')
                str_lst = data.split(' ')
                for s in str_lst:
                    if s.isdigit():
                        cell.append(int(s))
                # cell.append(data)
                # print(str_lst)
        if coordinates:
            Nodes_num += 1
            # if Nodes_num > 1:
            #     data = data.strip('\n')
            #     data = data[11:]
            #     str_lst = data.split(' ')
            #     for s in str_lst:
            #     #     if s.isdigit():
            #         point.append(float(s))
                # point.append(data)

            # try:
            #     test = int(data_list[0])
            # except ValueError:
            #     coordinates = False
        if facemt:
            face_num += 1
            if face_num > 0:
                data = data.strip('\n')
                data = data.strip('c')
                # data = data.strip('')
                str_lst = data.split(' ')
                for s in str_lst:
                    if s != '':
                        face.append(s)
                # data2 = data[:-2]
                # for i in range(len(data2)):
                #     if data2[i] ==
                # print(len(data2))
                # face.append(data)

            # try:
            #     test = int(data[0:16])
            # except ValueError:
            #     facemt = False


    Elements_num -= 1
    Nodes_num -= 1

    Nodes_New = np.zeros(Nodes_num * 3, dtype=np.float64).reshape(Nodes_num, 3)

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


    cell2 = np.array(cell).reshape(-1, 10)
    cell2 = np.delete(cell2, [0,1], axis=1)
    # point2 = np.array(point).reshape(-1, 3)
    face2 = []
    for i in range(len(face)):
        x = face[i].split(':')
        face2.append(int(x[0]))
        face2.append(int(x[1]))

    face2 = np.array(face2).reshape(-1,2)
    face3 = np.zeros((face2.shape[0], 4))
    for i in range(face2.shape[0]):
        for j in range(4):
            face3[i, j] = cell2[face2[i, 0]-1, canshu[face2[i, 1], j]-1] - 1


    b = cell2
    b = b - np.ones((b.shape[0], 8))


    return Nodes_New, b, face3
