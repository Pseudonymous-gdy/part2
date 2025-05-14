import numpy as np

# this is the easier to debug version of the code
def easy(M: int, N: int, K: int, T: int, A: np.ndarray, G: np.ndarray) -> int:
    """
    Parameters:
    M, N (int): Dimensions of the grid.
    K (int): Maximum allowed negative-reward cells.
    T (int): Initial gas amount.
    A (np.ndarray): Reward matrix (shape MxN).
    G (np.ndarray): Gas consumption matrix (shape MxN).
    Returns:
    int: Maximum total reward, or -1 if no valid path exists.
    """
    """
    Parameters:
    M, N (int): Dimensions of the grid.
    K (int): Maximum allowed negative-reward cells.
    T (int): Initial gas amount.
    A (np.ndarray): Reward matrix (shape MxN).
    G (np.ndarray): Gas consumption matrix (shape MxN).
    Returns:
    int: Maximum total reward, or -1 if no valid path exists.
    """
    # Implement your DP solution here
    def is_negative(a): 
        num=0
        for i in a:
            if i<0:
                num+=1
        return num

    dp=np.full((4,T+1,M,N),-float('inf')) 
    neg=np.zeros((4,T+1,M,N))

    for i in range(M):
        for j in range(N):
            for k in range(0,T+1):
                posk=k+G[i][j]
                negp=is_negative([A[i][j]])
                if posk>0 or (posk>=0 and i==M-1 and j==N-1):
                    if i>0 and j>0:
                        for n in range(4):
                            if neg[n][k][i-1][j]+negp<=K and dp[n][k][i-1][j]!=-float('inf'):
                                if n==0 and dp[1][posk][i][j]<dp[n][k][i-1][j]+A[i][j]:
                                    dp[1][posk][i][j]=dp[n][k][i-1][j]+A[i][j]
                                    neg[1][posk][i][j]=neg[n][k][i-1][j]+negp
                                elif (n==2 or n==3) and dp[0][posk][i][j]<dp[n][k][i-1][j]+A[i][j]:
                                    dp[0][posk][i][j]=dp[n][k][i-1][j]+A[i][j]
                                    neg[0][posk][i][j]=neg[n][k][i-1][j]+negp
                            if neg[n][k][i][j-1]+negp<=K and dp[n][k][i][j-1]!=-float('inf'):
                                if n==2 and dp[3][posk][i][j]<dp[n][k][i][j-1]+A[i][j]:
                                    dp[3][posk][i][j]=dp[n][k][i][j-1]+A[i][j]
                                    neg[3][posk][i][j]=neg[n][k][i][j-1]+negp
                                elif (n==0 or n==1) and dp[2][posk][i][j]<dp[n][k][i][j-1]+A[i][j]:
                                    dp[2][posk][i][j]=dp[n][k][i][j-1]+A[i][j]
                                    neg[2][posk][i][j]=neg[n][k][i][j-1]+negp
                    elif (i==0 and j==1) or (i==1 and j==0):
                        if i==0 and neg[0][k][0][0]+negp<=K and dp[0][k][0][0]!=-float('inf'):
                            dp[2][posk][i][j]=dp[0][k][0][0]+A[i][j]
                            neg[2][posk][i][j]=neg[0][k][0][0]+negp
                        elif i==1 and neg[0][k][0][0]+negp<=K and dp[0][k][0][0]!=-float('inf'):
                            dp[0][posk][i][j]=dp[0][k][0][0]+A[i][j]
                            neg[0][posk][i][j]=neg[0][k][0][0]+negp
                    elif i>0:
                        if neg[0][k][i-1][j]+negp<=K and dp[0][k][i-1][j]!=-float('inf') and dp[1][posk][i][j]<dp[0][k][i-1][j]+A[i][j]:
                            dp[1][posk][i][j]=dp[0][k][i-1][j]+A[i][j]
                            neg[1][posk][i][j]=neg[0][k][i-1][j]+negp
                    elif j>0:
                        if neg[2][k][i][j-1]+negp<=K and dp[2][k][i][j-1]!=-float('inf') and dp[3][posk][i][j]<dp[2][k][i][j-1]+A[i][j]:
                            dp[3][posk][i][j]=dp[2][k][i][j-1]+A[i][j]
                            neg[3][posk][i][j]=neg[2][k][i][j-1]+negp
                    else: 
                        if negp<=K and k==T+G[0][0]:  
                            dp[0][k][0][0]=A[0][0]
                            neg[0][k][0][0]=negp

    ans=-float('inf')
    for n in range(4):
        for x in range(T+1):
            if dp[n][x][M-1][N-1]>ans and neg[n][x][M-1][N-1]<=K:
                ans = dp[n][x][M-1][N-1]
    if ans==-float('inf'):
        ans=-1
    return int(ans)

# this is the harder to debug version of the code
def hard(M: int, N: int, K: int, T: int, A: np.ndarray, G: np.ndarray) -> int:
    """
    Parameters:
    M, N (int): Dimensions of the grid.
    K (int): Maximum allowed negative-reward cells.
    T (int): Initial gas amount.
    A (np.ndarray): Reward matrix (shape MxN).
    G (np.ndarray): Gas consumption matrix (shape MxN).
    Returns:
    int: Maximum total reward, or -1 if no valid path exists.
    """
    
    # Implement your DP solution here
    # Implement your DP solution here
    dp = [[[[[-float('inf'), 0] for _ in range(-2, 3)] for __ in range(K+1)] for ___ in range(N)] for ____ in range(M)]
    init_gas = T + G[0][0]


    first = 1 if A[0][0] < 0 else 0
    if first <= K and init_gas > 0:
        dp[0][0][first][0] = [A[0][0], init_gas]
        
    for i in range(M):
        for j in range(N):
            for k in range(K + 1):
                for t in range(-2, 3):
                    curr_reward, curr_gas = dp[i][j][k][t]
                    if curr_reward == -float('inf'):
                        continue
                    will_be_final = (i == M - 1 and j == N - 2) or (i == M - 2 and j == N - 1)
                    #right move
                    if j + 1 < N and t != 2:
                        new_gas = curr_gas + G[i][j + 1]
                        if (new_gas >= 0 and will_be_final) or (new_gas > 0 and not will_be_final):                            
                            new_k = k + (1 if A[i][j + 1] < 0 else 0)
                            if new_k <= K:
                                new_reward = curr_reward + A[i][j + 1]
                                if t == 0 or t == 1:
                                    new_t = t + 1
                                elif t == -1 or t == -2:
                                    new_t = 1
                                old_reward, old_gas = dp[i][j + 1][new_k][new_t]
                                if new_reward > old_reward or (new_reward == old_reward and new_gas > old_gas):
                                    dp[i][j+1][new_k][new_t] = [new_reward, new_gas]
                    #down move
                    if i + 1 < M and t != -2:
                        new_gas = curr_gas + G[i + 1][j]
                        if (new_gas >= 0 and will_be_final) or (new_gas > 0 and not will_be_final): 
                            new_k = k + (1 if A[i + 1][j] < 0 else 0)
                            if new_k <= K:
                                new_reward = curr_reward + A[i + 1][j]
                                if t == 0 or t == -1:
                                    new_t = t -1
                                elif t == 1 or t == 2:
                                    new_t = -1
                                old_reward, old_gas = dp[i + 1][j][new_k][new_t]
                                if new_reward > old_reward or (new_reward == old_reward and new_gas > old_gas):
                                    dp[i+1][j][new_k][new_t] = [new_reward, new_gas]



    max_reward = -float('inf')
    for k in range(K + 1):
        for t in range(-2, 3):
            curr_reward, curr_gas = dp[M - 1][N - 1][k][t]
            max_reward = max(max_reward, curr_reward)

    return max_reward if max_reward != -float('inf') else -1

