# Written Report
## 1. Introduction
This report is a written report of the project *part2*, which is a subsequent work of *part1*. This report will separately introduce the workflow of debugging with AI helpers (ChatGPT o3 + Canvas implementation, as is offered in AI Usage), and the final reflections with respect to debugging with AI.

## 2. Workflow of debugging `easy` with AI helpers
### 2.1. Preparation
To reach this work, I first read the code and identify certain bugs in the code. According to my observation, the bugs are mainly about the logic of the code and the data structure. Therefore, I've edited the code to make it easier for subsequent debugging with AI.

Basically, the problem lies in the design of negative reward tensor. Briefly, the code first recognized:
```python
neg=np.zeros((4,T+1,M,N))
```
And later the algorithm used codes similar to the following to update the negative reward tensor:
```python
neg[1][posk][i][j]=neg[n][k][i-1][j]+negp
```
This is errorneous because the cumulative sum of negative reward tensor is more determined by the path taken, rather than the position of the agent. Also, such design leaves out comparison between different paths of the same cumulative reward (and same gas consumption), making it difficult to determine the best path to maximize the reward. Therefore, I changed the design of negative reward tensor to tensor of dictionaries, like the code in the following:
```python
dp  = np.full((4, T + 1, M, N), -float("inf"))
neg = np.empty((4, T + 1, M, N), dtype=object)
for d in range(4):
    for g in range(T + 1):
        for i in range(M):
            for j in range(N):
                neg[d][g][i][j] = {}
```
In this way, the negative reward tensor is more flexible and can be used to compare different paths of the same cumulative reward (and same gas consumption).

### 2.2. Debugging with AI helpers
After the preparation, I started to debug the code with AI helpers. The edited code is provided, and to make it easier for AI helpers to understand and analyze, I also provided a brief introduction to the problem the code is trying to solve, and possible test cases (as follows).
```python
if __name__ == "__main__":
    # design a test case to test your solution
    M, N, K, T = 2, 2, 1, 9
    A = np.array([[1, 0], [0, 1]])
    G = np.array([[-1, -1], [-1, -1]])
    print(solve(M, N, K, T, A, G) == 2) # expected output: 2
    # a more difficult test case
    M, N, K, T = 3, 3, 1, 9
    A = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    G = np.array([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]])
    print(solve(M, N, K, T, A, G) == 3) # expected output: 3
    # much more difficult test case with exact numbers
    M, N, K, T = 16, 19, 5, 70
    A = np.array([[5, 3, -2, 4, 0, 2, 3, 3, -1, 4, 2, 1, 0, 6, -3, 5, 2, 4, 7],
                [-1, 0, 4, -2, 5, 3, 2, -1, 0, 2, 3, -2, 4, 1, 2, 0, -3, 5, 1],
                [3, -2, 1, 0, 4, -1, 5, 2, 3, 0, -4, 2, 6, -2, 0, 1, 4, 3, 2],
                [0, 4, 3, -3, 2, 1, 0, 2, 4, 3, 1, -2, 5, 0, 3, -1, 2, 4, 3],
                [6, 0, 2, 3, 0, -2, 4, 1, 3, 2, 5, 2, -1, 4, 3, 0, 2, -3, 4],
                [2, 3, -1, 4, 3, 5, -2, 4, 1, 3, 2, 0, 4, -2, 1, 2, 3, 3, 0],
                [-2, 4, 0, 1, 3, 2, 4, -3, 2, 5, 0, 3, 1, 2, 4, 0, 3, 2, 1],
                [1, 0, 3, 2, -1, 4, 2, 3, -2, 1, 4, 3, 0, 2, 1, -3, 4, 1, 3],
                [3, 2, 1, 4, 2, 0, 5, -1, 2, 3, -2, 1, 4, 0, 2, 3, 1, -4, 2],
                [0, 2, 3, -1, 4, 2, 1, 0, 3, 2, 4, 1, 2, 3, -2, 4, 0, 1, 3],
                [4, 3, -2, 0, 1, 2, 3, 4, -1, 2, 5, 3, 0, 1, 2, 3, 4, 0, 1],
                [2, 1, 0, 3, 2, -3, 4, 1, 0, 2, 3, 5, -1, 4, 2, 0, 3, 1, 4],
                [3, -2, 1, 4, 0, 3, 2, 1, -1, 2, 3, 4, 1, 0, 3, 2, 4, -2, 1],
                [0, 3, 2, 1, 4, -1, 3, 2, 0, 5, 1, 2, -2, 3, 4, 1, 0, 2, 3],
                [4, 0, 3, -1, 2, 4, 1, 3, 2, -2, 1, 4, 3, 0, 2, 1, 3, 2, -1],
                [5, 2, 4, 1, 3, 0, 2, 1, 4, 3, -2, 1, 0, 4, 2, 3, 1, 0, 10]])
    G = np.array([[0, -1, -2, -1, 0, -1, 0, -2, -1, -1, 0, -1, 0, -2, -2, -1, 0, -1, -1],
                [-1, 0, -1, -2, -1, -1, 0, -1, -2, -1, -1, 0, -1, -2, -1, -1, -1, -2, 0],
                [-1, -2, 0, -1, -1, 0, -2, -1, -1, -1, -2, 0, -1, -1, -2, -1, 0, -1, -1],
                [0, -1, -1, -2, 0, -1, -1, -1, -2, -1, 0, -1, -2, -1, -1, 0, -1, -2, -1],
                [-2, -1, 0, -1, -1, -2, -1, 0, -1, -1, -1, -2, -1, 0, -1, -1, -2, -1, 0],
                [-1, -1, -2, 0, -1, -1, 0, -1, -1, -2, -1, -1, -2, 0, -1, -1, -1, -2, -1],
                [0, -1, -1, -1, -2, -1, 0, -1, -1, -1, -1, -2, -1, -1, 0, -1, -2, -1, -1],
                [-1, -2, 0, -1, -1, -1, -2, -1, -1, 0, -1, -1, -1, -2, -1, 0, -1, -1, -2],
                [-1, 0, -1, -1, -2, -1, -1, -1, -2, -1, 0, -1, -1, -1, -2, -1, 0, -1, -2],
                [-2, -1, -1, 0, -1, -1, -1, -2, -1, -1, 0, -1, -1, -1, -2, -1, -1, -1, 0],
                [-1, -1, 0, -1, -1, -2, -1, -1, 0, -1, -1, -2, -1, -1, -1, -1, -2, -1, -1],
                [0, -1, -1, -2, -1, -1, -2, -1, 0, -1, -1, -1, -2, -1, 0, -1, -1, -1, -2],
                [-1, -2, -1, 0, -1, -1, -1, -2, -1, -1, -1, -2, 0, -1, -1, -1, -2, -1, -1],
                [-1, -1, -1, -2, -1, 0, -1, -1, -2, -1, -1, 0, -1, -1, -2, -1, -1, -1, -2],
                [0, -1, -1, -1, -2, -1, -1, -1, -1, 0, -1, -1, -1, -2, -1, -1, -1, -1, 0],
                [-1, -1, -2, -1, 0, -1, -1, -2, -1, -1, -1, -1, -2, -1, -1, -1, -1, 0, -1]])
    print("test case 1:", solve(M, N, K, T, A, G) == 94) # expected output: 94
    # new test case

    M, N, K, T = 18, 15, 8, 50

    A = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 20, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, -5, -5, -5, 15, 15, 15, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, -5, -5, -5, -8, -8, -8, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, -5, -5, -5, -8, -8, 20, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, -8, -8, -8, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, -8, -8, -8, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 20, 1, 1, -8, -8, -8, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, -8, -8, -8, 30, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 15, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 15, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 15, 1, 1, 20, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -10, -10, -10, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -10, -10, -10, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 20, -10, -10, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -10, -10, -10, 40, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -15, 1]
    ])

    G = np.array([
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -8, -8, -8, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -8, -8, -8, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -8, -8, -8, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -15, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -10, -10, -10, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -10, -10, -10, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -10, -20, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -5, -25, -1]
    ])
    print("test case 2:", solve(M, N, K, T, A, G) == 140)  # expected output: 140
    # test case 3
    M, N, K, T = 20, 20, 5, 100
    A = np.array([
        [4, 2, 1, 3, 4, 1, 5, 1, 2, 2, 3, 0, -2, 2, 1, 0, 1, 3, 2, 4],
        [3, 1, 2, 3, 2, 1, 0, 4, 3, 2, -1, 2, 1, 0, 1, 1, 2, 3, 4, 5],
        [1, 0, 2, 4, 3, 2, 1, 0, 1, 3, 2, 1, 0, 1, 1, 0, 2, 3, 1, 2],
        [0, -1, 3, 2, 1, 0, -2, 1, 2, 3, 2, 1, 2, 3, 1, 0, 1, 0, 2, 1],
        [2, 1, 1, 2, 0, 1, 1, 3, 2, 1, 0, 0, 1, 1, 2, 3, 0, -1, 2, 1],
        [1, 2, 3, 1, 2, 0, 1, 2, 1, 0, 1, 1, 0, 1, 2, 3, 1, 1, 1, 1],
        [2, 1, 0, 1, 2, 3, 1, 0, -1, 1, 2, 3, 1, 0, 1, 1, 0, 1, 2, 1],
        [1, 1, 2, 0, 1, 3, 2, 1, 0, 1, 2, 3, 1, 0, 1, 1, 2, 1, 0, 1],
        [1, 0, 1, 1, 2, 3, 2, 1, 0, 1, 1, 0, 1, 2, 1, 1, 0, 2, 3, 1],
        [2, 1, 2, 1, 0, 1, 3, 2, 1, 0, 2, 1, 0, 1, 2, 3, 1, 0, 1, 1],
        [1, 0, 1, 2, 1, 0, 1, 1, 2, 3, 1, 1, 2, 1, 0, 1, 1, 0, 2, 1],
        [1, 2, 3, 1, 1, 0, 1, 1, 2, 1, 0, 2, 3, 1, 0, 1, 2, 3, 1, 0],
        [0, 1, 2, 1, 1, 2, 1, 0, 1, 2, 3, 1, 1, 0, 1, 2, 3, 1, 0, 1],
        [1, 2, 1, 0, 2, 3, 1, 0, 1, 2, 3, 1, 1, 0, 1, 2, 3, 1, 0, 1],
        [1, 1, 2, 3, 1, 0, 1, 1, 2, 3, 1, 0, 1, 2, 1, 0, 1, 2, 3, 1],
        [0, 1, 2, 3, 1, 0, 1, 1, 2, 3, 1, 0, 2, 1, 0, 1, 2, 3, 1, 1],
        [1, 2, 1, 0, 1, 1, 2, 3, 1, 0, 1, 2, 3, 1, 0, 1, 1, 2, 1, 0],
        [1, 0, 1, 1, 2, 3, 1, 0, 1, 2, 3, 1, 0, 1, 2, 3, 1, 1, 0, 1],
        [1, 1, 2, 3, 1, 0, 1, 2, 3, 1, 0, 1, 1, 2, 3, 1, 0, 1, 1, 0],
        [1, 0, 1, 2, 3, 1, 0, 1, 2, 1, 0, 1, 2, 3, 1, 0, 1, 2, 3, 1]
    ])
    G = np.array([
        [-1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2],
        [-1, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -1, -2, -1, -1],
        [-1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1],
        [-1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1],
        [-1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2],
        [-1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1],
        [-1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2],
        [-1, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1],
        [-1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1],
        [-1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2],
        [-1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1],
        [-1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2],
        [-1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2],
        [-1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1],
        [-1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2],
        [-1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1],
        [-1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2],
        [-1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1],
        [-1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2],
        [-1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1]
    ])
    # Replace <expected_value> with the correct expected output if known.
    print("test case 3:", solve(M, N, K, T, A, G) == 77)
    # new test case
    M, N, K, T = 5, 5, 3, 100
    A = np.array([
        [1, -1, 2, 4, 5],
        [1, -1, 2, 3, 4],
        [2, 3, 4, 100, 6],
        [4, 5, 6, 7, 8],
        [5, 6, 7, 8, 9]
    ])
    G = np.array([
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1]
    ])
    print("test case 4:", solve(M, N, K, T, A, G) == 132)
    # new test case
    M, N, K, T = 20, 20, 30, 100
    A = np.array([
        [4, 2, 1, 3, 4, 1, 5, 1, 2, 2, 3, 0, -2, 2, 1, 0, 1, 3, 2, 4],
        [3, 1, 2, 3, 2, 1, 0, 4, 3, 2, -1, 2, 1, 0, 1, 1, 2, 3, 4, 5],
        [1, 0, 2, 4, 3, 2, 1, 0, 1, 3, 2, 1, 0, 1, 1, 0, 2, 3, 1, 2],
        [0, -1, 3, 2, 1, 0, -2, 1, 2, 3, 2, 1, 2, 3, 1, 0, 1, 0, 2, 1],
        [2, 1, 1, 2, 0, 1, 1, 3, 2, 1, 0, 0, 1, 1, 2, 3, 0, -1, 2, 1],
        [1, 2, 3, 1, 2, 0, 1, 2, 1, 0, 1, 1, 0, 1, 2, 3, 1, 1, 1, 1],
        [2, 1, 0, 1, 2, 3, 1, 0, -1, 1, 2, 3, 1, 0, 1, 1, 0, 1, 2, 1],
        [1, 1, 2, 0, 1, 3, 2, 1, 0, 1, 2, 3, 1, 0, 1, 1, 2, 1, 0, 1],
        [1, 0, 1, 1, 2, 3, 2, 1, 0, 1, 1, 0, 1, 2, 1, 1, 0, 2, 3, 1],
        [2, 1, 2, 1, 0, 1, 3, 2, 1, 0, 2, 1, 0, 1, 2, 3, 1, 0, 1, 1],
        [1, 0, 1, 2, 1, 0, 1, 1, 2, 3, 1, 1, 2, 1, 0, 1, 1, 0, 2, 1],
        [1, 2, 3, 1, 1, 0, 1, 1, 2, 1, 0, 2, 3, 1, 0, 1, 2, 3, 1, 0],
        [0, 1, 2, 1, 1, 2, 1, 0, 1, 2, 3, 1, 1, 0, 1, 2, 3, 1, 0, 1],
        [1, 2, 1, 0, 2, 3, 1, 0, 1, 2, 3, 1, 1, 0, 1, 2, 3, 1, 0, 1],
        [1, 1, 2, 3, 1, 0, 1, 1, 2, 3, 1, 0, 1, 2, 1, 0, 1, 2, 3, 1],
        [0, 1, 2, 3, 1, 0, 1, 100, 2, 3, 1, 0, 2, 1, 0, 1, 2, 3, 1, 1],
        [1, 2, 1, 0, 1, 1, 2, 3, 1, 0, 1, 2, 3, 1, 0, 1, 1, 2, 1, 0],
        [1, 0, 1, 1, 2, 3, 1, 0, 1, 2, 3, 1, 0, 1, 2, 3, 1, 1, 0, 1],
        [1, 1, 2, 3, 1, 0, 1, 2, 3, 1, 0, 1, 1, 2, 3, 1, 0, 1, 1, 0],
        [1, 0, 1, 2, 3, 1, 0, 1, 2, 1, 0, 1, 2, 3, 1, 0, 1, 2, 3, 1]
    ])
    G = np.array([
        [-1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2],
        [-1, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -1, -2, -1, -1],
        [-1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1],
        [-1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1],
        [-1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2],
        [-1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1],
        [-1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2],
        [-1, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1],
        [-1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1],
        [-1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2],
        [-1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1],
        [-1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2],
        [-1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2],
        [-1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1],
        [-1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2],
        [-1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1],
        [-1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2],
        [-1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1],
        [-1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -2],
        [-1, -2, -1, -1, -2, -1, -1, -2, -1, -1, -1, -2, -1, -1, -2, -1, -1, -2, -1, -1]
    ])
    print("test case 5:", solve(M, N, K, T, A, G) == 77)
    # new test case
    M, N, K, T = 6, 6, 9, 34
    A = np.array([
        [198, 22, -87, -6, -53, -86],
        [99, 26, 72, -19, 10, -48],
        [-77, 53, 164, 42, -60, 56],
        [-86, -56, -36, 96, -30, -92],
        [-13, 28, 35, 92, 54, 172],
        [35, 62, 62, -68, 22, 70]
    ])
    G = np.array([
        [-18, -7, -12, -14, -12, -13],
        [-9, -6, -1, -5, -16, -18],
        [-9, -13, -1, 0, -18, -16],
        [-6, -7, -18, 0, -16, -7],
        [-14, -12, -6, 0, 0, 0],
        [-2, -14, -4, -1, -17, 0],
    ])
    print("test case 6:", solve(M,N,K,T,A,G) == 1008)
    # new test case
    M, N, K, T = 4, 3, 0, 0
    A = np.array([
        [1, -2, 3],
        [4, -5, 6],
        [7, -8, 9],
        [10, 11, 12]
    ])
    G = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    print("test case 7:", solve(M,N,K,T,A,G)==-1)
    # new test case
    M, N, K, T = 2, 1, 4, 32
    A = np.array([[190], [174]])
    G = np.array([[-6],[-23]])
    print("test case 8:", solve(M,N,K,T,A,G)==364)
    M, N, K, T = 7, 10, 9, 84
    A = np.array([
        [ -3,  87,  91, -13,   8,  62, -31,  57,  98, -27],
        [-38,  55, -41, -22, -64, -79,  95,  47,  -5, -33],
        [ 91, -84, -89,  49,  44,  37,  17,  92, -79,  19],
        [-51, -87,  91,  74, -72,  63,  79,  24, -94, -72],
        [ 38, -68,  33,  10, -22,  59,  78, -43, -75,  73],
        [-72, -97,  -2,  40, -62, -86,  18, -72,  56, -26],
        [ 15, -69,  29,  74, -17, -79, -74,  94, -20,  25]
        ])
    G = np.array([
        [ -9,   0,  -6,  -5, -16,  -1, -15,  -2, -16, -18],
        [-20, -15,  -6, -19, -18,  -4,  -9,  -3,  -6,  -9],
        [ -5, -13,  -7,  -4,   0, -13,  -6,  -7,  -3,  -1],
        [-14, -18,  -9,   0, -18,  -1,  -7,  -6,  -7, -14],
        [ -4,  -9,  -1, -17, -12, -15,  -5,   0,   0, -19],
        [-13, -10,  -8,  -4, -13,   0,  -3,  -3,  -7,  -6],
        [-15, -10,  -8, -20, -15, -12, -13, -10,  -2, -12]
        ])
    print("test case 9:", solve(M,N,K,T,A,G)==229)
```

AI debugging offered a correct solution, and reported several issues in the code:
- Bugs when operating dictionaries: the code did not successfully handle the way to operate dictionaries
- Gas index out of range: when gas_after < 0, the code operated incorrectly on last but several gas indexs.
- Ignorance of better paths: the code did not consider the possibility of better paths with same reward but less negative reward.
- Minor Issues, like shape error, incomplete code to solve the problems above, etc.

## 3. Workflow of debugging `hard` with AI helpers
### 3.1. Preparation
The code was viewed and executed before debugging, with the result of 40+/53 passed test cases. It was also tested using my own test cases, which were left with only 1 error, which could be debugged without preprocessing.

### 3.2. Debugging

The code was debugged using AI helpers, with the following reported errors:

- Edge cases: Goal-exception logic wasn't mirrored for both directions.
- Will_be_final: The code did not correctly handle the case where the goal is not reached (gas = 0 when not final).
- Minor issues, like memory blow-up when storing lists of reward-gas pairs, etc.

## 4. Summary and Reflections
### 4.1. Strengths of AI debugging
AI debugging was capable of identifying and fixing issues in the code if given test cases to study, including bugs in the logic and edge cases. It was also able to identify and fix minor issues, such as memory blow-up and shape errors.

### 4.2. Useful prompts
- When debugging, it is important to provide a clear and concise description of the problem, including the expected output and any relevant test cases.
- It is also important to provide the code to be debugged, along with any relevant information about the problem.
- Offering test cases as examples can help AI helpers to understand the problem better and provide more accurate solutions.

### 4.3. Limitations of AI debugging
AI debugging is not perfect when conducting given instructions (for example, it would like to edit the code unexpectedly like changing codes in unrelated parts, or in its own style of variables and functions, etc.), and in most of the time, it is possible to get a wrong solution with undetectable errors (for instance, it would leave a bug in the code without noticing that the test cases still failed, etc.).