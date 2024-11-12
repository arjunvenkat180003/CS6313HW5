import numpy as np
with open("Observations.txt", "r") as my_file:
    data = my_file.read() 

data = data.split("\n")

data = [x.split(' ') for x in data]

#print(data)

for i in range(len(data)):
    for j in range(len(data[0])):
        data[i][j] = float(data[i][j])

#print(data)

y = [row[2] for row in data]
X = [[1]+row[:2] for row in data]
#print('x', X)

y = np.asarray(y)
X = np.asarray(X)

first_term = np.linalg.inv(np.matmul(np.transpose(X), X))
second_term = np.matmul(np.transpose(X), y)
slopes = np.matmul(first_term, second_term)

print('slopes', slopes)

print('x', X)
print('y    ', y)
y_hat_np = np.matmul(X, slopes)
print('y_hat', y_hat_np)
