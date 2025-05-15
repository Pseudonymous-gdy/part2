import numpy as np
from typing import Dict, List
# this is the easier to debug version of the code
def easy(M: int, N: int, K: int, T: int, A: np.ndarray, G: np.ndarray) -> int:
    """Dynamic‑programming solver for the *easy* variant of the
    "max‑reward path with gas/direction/neg‑cell constraints".

    The implementation keeps the original author’s 4‑direction / 2‑step‐streak
    encoding but fixes the earlier *IndexError* and now treats reward ties
    consistently in **both** directions (↓ and →).
    """

    # -------------------------------------------------------------- helpers
    def is_neg(val: int) -> int:  # 1 iff cell reward is negative
        return 1 if val < 0 else 0

    # -------------------------------------------------------------- allocate DP
    dp  = np.full((4, T + 1, M, N), -float("inf"))
    neg = np.empty((4, T + 1, M, N), dtype=object)
    for d in range(4):
        for g in range(T + 1):
            for i in range(M):
                for j in range(N):
                    neg[d][g][i][j] = {}

    # -------------------------------------------------------------- seed (0,0)
    start_gas = T + G[0, 0]
    if (start_gas > 0 or (start_gas == 0 and M == 1 and N == 1)) \
            and is_neg(A[0, 0]) <= K and 0 <= start_gas <= T:
        # allow first move to be either ↓ or → (streak‑1)
        for d0 in (0, 2):
            dp[d0][start_gas][0][0] = A[0, 0]
            neg[d0][start_gas][0][0][A[0, 0]] = is_neg(A[0, 0])

    # -------------------------------------------------------------- main sweep
    for i in range(M):
        for j in range(N):
            if i == j == 0:
                continue
            for gas_before in range(T + 1):
                for d_prev in range(4):
                    # ───────────────────────── move ↓ ─────────────────────────
                    if i > 0 and dp[d_prev][gas_before][i - 1][j] != -float("inf"):
                        gas_after = gas_before + G[i, j]
                        if gas_after < 0 or gas_after > T or (gas_after == 0 and (i != M - 1 or j != N - 1)):
                            pass
                        else:
                            if d_prev == 0:
                                d_new = 1  # second ↓
                            elif d_prev in (2, 3):
                                d_new = 0  # first ↓
                            else:  # d_prev == 1 (would be third ↓)
                                d_new = None
                            if d_new is not None:
                                base_reward = dp[d_prev][gas_before][i - 1][j]
                                used_neg    = neg[d_prev][gas_before][i - 1][j].get(base_reward, K + 1)
                                total_neg   = used_neg + is_neg(A[i, j])
                                new_reward  = base_reward + A[i, j]
                                if total_neg <= K and (
                                        new_reward > dp[d_new][gas_after][i][j] or
                                        (new_reward == dp[d_new][gas_after][i][j] and
                                         total_neg < neg[d_new][gas_after][i][j].get(new_reward, K + 1))):
                                    dp[d_new][gas_after][i][j] = new_reward
                                    neg[d_new][gas_after][i][j][new_reward] = total_neg

                    # ───────────────────────── move → ─────────────────────────
                    if j > 0 and dp[d_prev][gas_before][i][j - 1] != -float("inf"):
                        gas_after = gas_before + G[i, j]
                        if gas_after < 0 or gas_after > T or (gas_after == 0 and (i != M - 1 or j != N - 1)):
                            pass
                        else:
                            if d_prev == 2:
                                d_new = 3  # second →
                            elif d_prev in (0, 1):
                                d_new = 2  # first →
                            else:  # d_prev == 3 (would be third →)
                                d_new = None
                            if d_new is not None:
                                base_reward = dp[d_prev][gas_before][i][j - 1]
                                used_neg    = neg[d_prev][gas_before][i][j - 1].get(base_reward, K + 1)
                                total_neg   = used_neg + is_neg(A[i, j])
                                new_reward  = base_reward + A[i, j]
                                if total_neg <= K and (
                                        new_reward > dp[d_new][gas_after][i][j] or
                                        (new_reward == dp[d_new][gas_after][i][j] and
                                         total_neg < neg[d_new][gas_after][i][j].get(new_reward, K + 1))):
                                    dp[d_new][gas_after][i][j] = new_reward
                                    neg[d_new][gas_after][i][j][new_reward] = total_neg

    # -------------------------------------------------------------- extract ans
    best = -float("inf")
    for d in range(4):
        for g in range(T + 1):
            reward = dp[d][g][M - 1][N - 1]
            if reward != -float("inf") and neg[d][g][M - 1][N - 1].get(reward, K + 1) <= K:
                best = max(best, reward)

    return -1 if best == -float("inf") else int(best)

# this is the harder to debug version of the code
def hard(M: int, N: int, K: int, T_init: int, A: np.ndarray, G: np.ndarray) -> int:
    """Dynamic‑programming solver for the *hard* variant of the problem.

    We maintain for each grid cell `(i, j)`, negative‑count `k ∈ [0,K]`, and
    direction‑streak state `t ∈ {‑2,‑1,0,1,2}` *a dictionary* mapping
    **remaining gas → maximum reward**.  A Pareto filter in `_update_state`
    prevents exponential blow‑up while preserving optimality.
    """
    T_TO_IDX = {t: t + 2 for t in range(-2, 3)}      # ‑2→0, ‑1→1, 0→2, 1→3, 2→4
    IDX_TO_T = {idx: t for t, idx in T_TO_IDX.items()}
    
    
    def _update_state(dest: Dict[int, int], gas: int, reward: int) -> None:
        """Insert (gas → reward) into *dest* dict while keeping only
        Pareto‑optimal entries.
    
        * For a fixed gas, keep the highest reward.
        * Drop entries that are *dominated* (i.e. another entry has ≥reward and ≥gas).
        """
        # If current gas already recorded with >= reward, nothing to do
        if gas in dest and dest[gas] >= reward:
            return
    
        # Insert / overwrite
        dest[gas] = reward
    
        # Prune dominated states to keep dicts small
        keys: List[int] = list(dest.keys())
        for g in keys:
            if g == gas:
                continue
            r = dest[g]
            # if (g, r) is dominated by (gas, reward) → remove
            if g <= gas and r <= reward:
                del dest[g]
            # if new state is dominated by an existing one → remove new & exit
            elif g >= gas and r >= reward and (g != gas or r != reward):
                del dest[gas]
                break
    # DP structure:  M × N × (K+1) × 5  where each leaf is {gas → reward}
    dp: List[List[List[List[Dict[int, int]]]]] = [
        [
            [[{} for _ in range(5)] for _ in range(K + 1)]
            for _ in range(N)
        ]
        for _ in range(M)
    ]

    # ------------------------------------------------------------------
    # Seed at (0,0)
    # ------------------------------------------------------------------
    init_gas = T_init + G[0][0]  # pay cost of start cell
    if init_gas <= 0:
        return -1  # cannot start the journey

    init_k = 1 if A[0][0] < 0 else 0
    if init_k > K:
        return -1

    # t = 0 means "no prior direction" (streak length 0)
    dp[0][0][init_k][T_TO_IDX[0]][init_gas] = A[0][0]

    # ------------------------------------------------------------------
    # Forward DP scan (row‑major, moves only right or down)
    # ------------------------------------------------------------------
    for i in range(M):
        for j in range(N):
            for k in range(K + 1):
                for t_idx in range(5):
                    t = IDX_TO_T[t_idx]
                    state: Dict[int, int] = dp[i][j][k][t_idx]
                    if not state:
                        continue

                    # Flags to allow finishing with gas ≥ 0 at goal
                    right_is_final = (i == M - 1 and j == N - 2)
                    down_is_final = (i == M - 2 and j == N - 1)

                    for curr_gas, curr_reward in state.items():
                        # -----------------------------
                        # Move RIGHT
                        # -----------------------------
                        if j + 1 < N and t != 2:  # cannot exceed 2‑step streak
                            new_gas = curr_gas + G[i][j + 1]
                            # Gas must stay >0 except possibly *at* goal cell
                            if (right_is_final and new_gas >= 0) or (not right_is_final and new_gas > 0):
                                new_k = k + (1 if A[i][j + 1] < 0 else 0)
                                if new_k <= K:
                                    new_reward = curr_reward + A[i][j + 1]
                                    new_t = t + 1 if t in (0, 1) else 1  # adjust streak
                                    _update_state(dp[i][j + 1][new_k][T_TO_IDX[new_t]], new_gas, new_reward)

                        # -----------------------------
                        # Move DOWN
                        # -----------------------------
                        if i + 1 < M and t != -2:  # cannot exceed 2‑step streak
                            new_gas = curr_gas + G[i + 1][j]
                            if (down_is_final and new_gas >= 0) or (not down_is_final and new_gas > 0):
                                new_k = k + (1 if A[i + 1][j] < 0 else 0)
                                if new_k <= K:
                                    new_reward = curr_reward + A[i + 1][j]
                                    new_t = t - 1 if t in (0, -1) else -1  # adjust streak
                                    _update_state(dp[i + 1][j][new_k][T_TO_IDX[new_t]], new_gas, new_reward)

    # ------------------------------------------------------------------
    # Extract answer at goal cell (M‑1, N‑1) safely
    # ------------------------------------------------------------------
    best = -float('inf')
    goal_states = dp[M - 1][N - 1]
    for k in range(K + 1):
        for t_idx in range(5):
            cell = goal_states[k][t_idx]
            if cell:  # non‑empty dict
                cell_best = max(cell.values())
                if cell_best > best:
                    best = cell_best

    return best if best != -float('inf') else -1

