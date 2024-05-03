import numpy as np

EMPTY = 0

# Game State
ONGOING = 0
X_WON = 1
O_WON = 2
DRAW = 3

# Players
AI = -1
PLAYER = 1

# count
count = 0

tic_tac_toe_board = np.zeros((3, 3))


def is_victory(tic_tac_toe_board):

    # Check rows
    for row in range(3):
        if np.all(tic_tac_toe_board[row] == -1):
            return X_WON
        elif np.all(tic_tac_toe_board[row] == 1):
            return O_WON
    # Check columns
    for col in range(3):
        if np.all(tic_tac_toe_board[:, col] == -1):
            return X_WON
        elif np.all(tic_tac_toe_board[:, col] == 1):
            return O_WON
    # Check diagonals
    if np.all(np.diag(tic_tac_toe_board) == -1):
        return X_WON
    elif np.all(np.diag(tic_tac_toe_board) == 1):
        return O_WON
    elif np.all(np.diag(np.fliplr(tic_tac_toe_board)) == -1):
        return X_WON
    elif np.all(np.diag(np.fliplr(tic_tac_toe_board)) == 1):
        return O_WON
    # Check for draw
    if np.all(tic_tac_toe_board != EMPTY):
        return DRAW
    else:
        return ONGOING


def minimax(tic_tac_toe_board, depth, maxAgent):
    global count
    state = is_victory(tic_tac_toe_board)
    if state != ONGOING:
        if state == 1:
            return -1
        elif state == 2:
            return 1
        else:
            return 0
    if maxAgent:
        best_score = -np.inf
        for i in range(3):
            for j in range(3):
                if tic_tac_toe_board[i][j] == EMPTY:
                    tic_tac_toe_board[i][j] = AI
                    count = count + 1
                    current_score = minimax(tic_tac_toe_board, depth + 1, False)
                    tic_tac_toe_board[i][j] = EMPTY
                    if current_score > best_score:
                        best_score = current_score
        return best_score
    else:
        best_score = np.inf
        for i in range(3):
            for j in range(3):
                if tic_tac_toe_board[i][j] == EMPTY:
                    tic_tac_toe_board[i][j] = PLAYER
                    count = count + 1
                    current_score = minimax(tic_tac_toe_board, depth + 1, True)
                    tic_tac_toe_board[i][j] = EMPTY
                    if current_score < best_score:
                        best_score = current_score
        return best_score


def play_game():
    global count
    global tic_tac_toe_board
    tic_tac_toe_board = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    print(tic_tac_toe_board)
    if np.count_nonzero(tic_tac_toe_board) % 2 == 0:  # AI's move
        score = minimax(tic_tac_toe_board, 0, True)
    print("\n Number of node visited for this states: ", count)
    state = is_victory(tic_tac_toe_board)
    # print(state)
    if state == 1:
        print("Computer wins!")
    elif state == 2:
        print("Player wins!")


play_game()


import numpy as np

EMPTY = 0

# Game State
ONGOING = 0
X_WON = 1
O_WON = 2
DRAW = 3

# Players
AI = -1
PLAYER = 1

# count
count = 0

tic_tac_toe_board = np.zeros((3, 3))


def is_victory(tic_tac_toe_board):

    # Check rows
    for row in range(3):
        if np.all(tic_tac_toe_board[row] == -1):
            return X_WON
        elif np.all(tic_tac_toe_board[row] == 1):
            return O_WON
    # Check columns
    for col in range(3):
        if np.all(tic_tac_toe_board[:, col] == -1):
            return X_WON
        elif np.all(tic_tac_toe_board[:, col] == 1):
            return O_WON
    # Check diagonals
    if np.all(np.diag(tic_tac_toe_board) == -1):
        return X_WON
    elif np.all(np.diag(tic_tac_toe_board) == 1):
        return O_WON
    elif np.all(np.diag(np.fliplr(tic_tac_toe_board)) == -1):
        return X_WON
    elif np.all(np.diag(np.fliplr(tic_tac_toe_board)) == 1):
        return O_WON
    # Check for draw
    if np.all(tic_tac_toe_board != EMPTY):
        return DRAW
    else:
        return ONGOING


def minMaxwithpruning(tic_tac_toe_board, depth, alpha, beta, maxAgent):
    global count
    state = is_victory(tic_tac_toe_board)
    if state != ONGOING:
        if state == 1:
            return -1
        elif state == 2:
            return 1
        else:
            return 0
    if maxAgent:
        best_score = -np.inf
        for i in range(3):
            for j in range(3):
                if tic_tac_toe_board[i][j] == EMPTY:
                    tic_tac_toe_board[i][j] = AI
                    current_score = minMaxwithpruning(
                        tic_tac_toe_board, depth + 1, alpha, beta, False
                    )
                    count = count + 1
                    tic_tac_toe_board[i][j] = EMPTY
                    if current_score > best_score:
                        best_score = current_score
                    alpha = max(alpha, current_score)
                    if beta <= alpha:
                        break
        return best_score
    else:
        best_score = np.inf
        for i in range(3):
            for j in range(3):
                if tic_tac_toe_board[i][j] == EMPTY:
                    tic_tac_toe_board[i][j] = PLAYER
                    current_score = minMaxwithpruning(
                        tic_tac_toe_board, depth + 1, alpha, beta, True
                    )
                    count = count + 1
                    tic_tac_toe_board[i][j] = EMPTY
                    if current_score < best_score:
                        best_score = current_score
                    beta = min(beta, current_score)
                    if beta <= alpha:
                        break
        return best_score


def play_game():
    global count
    global tic_tac_toe_board
    tic_tac_toe_board = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    print(tic_tac_toe_board)
    if np.count_nonzero(tic_tac_toe_board) % 2 == 0:  # AI's move
        move = minMaxwithpruning(tic_tac_toe_board, 0, -np.inf, np.inf, True)
    print("\n Number of node visited for this states: ", count)
    state = is_victory(tic_tac_toe_board)
    if state == 1:
        print("Computer wins!")
    elif state == 2:
        print("Player wins!")


play_game()
