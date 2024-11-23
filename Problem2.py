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
y_hat = np.matmul(X, slopes)
print('y_hat', y_hat)

# Calculate the mean of the observed values (y)
y_mean = np.mean(y)

# Calculate SST (Total Sum of Squares)
SST = np.sum((y - y_mean) ** 2)

# Calculate SSR (Sum of Squares due to Regression)
SSR = np.sum((y_hat - y_mean) ** 2)

# Calculate SSE (Sum of Squares of Errors)
SSE = np.sum((y - y_hat) ** 2)

# Calculate Mean Squares
print(f'(X.shape[1] - 1: {X.shape[1] - 1}')
print(f'(X.shape[0] - X.shape[1]: {X.shape[0] - X.shape[1]}')

MSR = SSR / (X.shape[1] - 1)  
MSE = SSE / (X.shape[0] - X.shape[1])  

# Calculate F statistic
F = MSR / MSE

# Calculate R squared
R_squared = (SSR / SST) * 100


# Output the Sum of Squares and Mean Squares
print('SSR (Sum of Squares due to Regression):', SSR)
print('SSE (Sum of Squares of Errors):', SSE)
print('SST (Total Sum of Squares):', SST)
print('MSR (Mean Square due to Regression):', MSR)
print('MSE (Mean Square Error):', MSE)

# Output F statistic and R squared
print('F statistic:', F)
print('R squared as a percentage:', R_squared)