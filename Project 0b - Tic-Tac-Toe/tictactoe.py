"""
Tic Tac Toe Player
"""
import copy
import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    X_count = sum(row.count(X) for row in board)
    O_count = sum(row.count(O) for row in board)
    if X_count == O_count:
        return X
    return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    return set([
        (i, j)
        for i in range(3)
        for j in range(3)
        if board[i][j] == EMPTY
    ])


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    new_board = copy.deepcopy(board)
    if new_board[action[0]][action[1]] != EMPTY:
        raise Exception("Invalid action.")
    else:
        new_board[action[0]][action[1]] = player(new_board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    lines = [
        [board[i][0] for i in range(3)],
        [board[i][1] for i in range(3)],
        [board[i][2] for i in range(3)],
        board[0],
        board[1],
        board[2],
        [board[i][i] for i in range(3)],
        [board[2 - i][i] for i in range(3)],
    ]

    for line in lines:
        if line.count(X) == 3:
            return X
        if line.count(O) == 3:
            return O
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board):
        return True
    return all(EMPTY not in row for row in board)


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """

    def maxValue(board):
        if terminal(board):
            return utility(board)
        return max(minValue(result(board, action)) for action in actions(board))

    def minValue(board):
        if terminal(board):
            return utility(board)
        return min(maxValue(result(board, action)) for action in actions(board))

    if terminal(board):
        return None

    tourn = player(board)
    if tourn == X:
        value = -math.inf
        move = None
        for action in actions(board):
            minValueResult = minValue(result(board, action))
            if minValueResult > value:
                value = minValueResult
                move = action
    else:
        value = math.inf
        move = None
        for action in actions(board):
            maxValueResult = maxValue(result(board, action))
            if maxValueResult < value:
                value = maxValueResult
                move = action
    return move
