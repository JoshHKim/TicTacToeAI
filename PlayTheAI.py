import numpy as np
import os
import warnings
from os.path import dirname, join
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn.neural_network')
current_dir = dirname(__file__)
file_path = join(current_dir, "datasets/tictac_single.txt")

A = np.loadtxt(file_path)
np.random.shuffle(A)
X = A[:, :9]
y = A[:, 9:]
y=y.ravel()
mlp = MLPClassifier(random_state=42, max_iter=100)
mlp.fit(X, y)

def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 9)

def check_win(board, player):
    win_conditions = [
        [board[0][0], board[0][1], board[0][2]],
        [board[1][0], board[1][1], board[1][2]],
        [board[2][0], board[2][1], board[2][2]],
        [board[0][0], board[1][0], board[2][0]],
        [board[0][1], board[1][1], board[2][1]],
        [board[0][2], board[1][2], board[2][2]],
        [board[0][0], board[1][1], board[2][2]],
        [board[2][0], board[1][1], board[0][2]],
    ]
    return [player, player, player] in win_conditions

def check_tie(board):
    for row in board:
        if " " in row:
            return False
    return True

def get_player_move(board):
    while True:
        try:
            row = int(input("Player X, enter your move row (1-3): ")) - 1
            col = int(input("Player X, enter your move column (1-3): ")) - 1
            if 0 <= row <= 2 and 0 <= col <= 2 and board[row][col] == " ":
                return row, col
            else:
                print("Invalid move. Make sure your spot is not already taken and is within the 1-3 range.")
        except ValueError:
            print("Invalid input. Please enter numbers only.")

def convert_board(board):
    np_board = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                np_board.append(0)
            elif board[i][j] == 'X':
                np_board.append(1)
            else:
                np_board.append(-1)
    return np.asarray(np_board)

def get_program_move(board):
    X_board = convert_board(board)
    y_board = mlp.predict(X_board.reshape(1,9))
    y_move = (int)(y_board[0])
    return [y_move//3, y_move%3]

def play_game():
    board = [[" " for _ in range(3)] for _ in range(3)]
    current_player = "X"
    
    while True:
        print_board(board)
        
        if current_player == "X":
            row, col = get_player_move(board)
        else:
            print("Program is making a move...")
            row, col = get_program_move(board)
        
        board[row][col] = current_player
        
        if check_win(board, current_player):
            print_board(board)
            print(f"Player {current_player} wins!" if current_player == "X" else "Program wins!")
            break
        elif check_tie(board):
            print_board(board)
            print("It's a tie!")
            break
        
        current_player = "O" if current_player == "X" else "X"

if __name__ == "__main__":
    play_game()
