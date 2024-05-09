import numpy as np

pod_result = np.loadtxt('pod_result.txt')

result = np.loadtxt('result300.txt')

delta = np.abs(pod_result-result)

for i in range(pod_result.shape[1]):
    ans = np.max(delta[:, i])
    ans2 = np.min(delta[:, i])
    ans3 = np.mean(delta[:, i])

    # print(i)
    print(f'第{i}步最大偏差：{ans},最小误差：{ans2},平均误差：{ans3}')
