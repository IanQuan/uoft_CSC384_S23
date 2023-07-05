import argparse
import copy
import sys
import time


class State:
    def __init__(self, board, curr_turn, parent=None):
        """
        Assume the board is 8x8
        :param board:
        :param curr_turn: Either ['r', 'R'] or ['b', 'B']
        :param parent: Parent state on the game search tree
        """

        self.board = board
        self.curr_turn = curr_turn
        self.parent = parent

    def __str__(self) -> str:
        """
        :return: String representation of the board (Used as the key of the state caching dictionary)
        """
        result = ''
        for row in self.board:
            for col in row:
                result += col
        return result

    def display(self):
        for i in self.board:
            for j in i:
                print(j, end="")
            print("")
        print("")

    def check_move(self, coord, direction):
        """
        Check if the given direction of move is possible. If yes, return the next state
        :param coord: Coordinates of the piece on the board that we want to move
        :param direction: Direction of the move
        :return: A tuple: (True, new_state) or (False, None)
        """
        movable = False
        y = coord[0]
        x = coord[1]
        piece = self.board[y][x]  # piece can be 'b', 'r', 'B' or 'R'

        if direction == 'up_right':
            if y == 0 or x == 7: return movable, None  # check if the move will be out of bound
            if self.board[y - 1][x + 1] == '.':
                movable = True
                new_state = State(copy.deepcopy(self.board), get_opp_char(piece))
                new_state.board[y - 1][x + 1] = piece
                new_state.board[y][x] = '.'
                return movable, new_state
        elif direction == 'up_left':
            if y == 0 or x == 0: return movable, None  # check if the jump will be out of bound
            if self.board[y - 1][x - 1] == '.':
                movable = True
                new_state = State(copy.deepcopy(self.board), get_opp_char(piece))
                new_state.board[y - 1][x - 1] = piece
                new_state.board[y][x] = '.'
                return movable, new_state
        elif direction == 'down_right':
            if y == 7 or x == 7: return movable, None  # check if the jump will be out of bound
            if self.board[y + 1][x + 1] == '.':
                movable = True
                new_state = State(copy.deepcopy(self.board), get_opp_char(piece))
                new_state.board[y + 1][x + 1] = piece
                new_state.board[y][x] = '.'
                return movable, new_state
        elif direction == 'down_left':
            if y == 7 or x == 0: return movable, None  # check if the jump will be out of bound
            if self.board[y + 1][x - 1] == '.':
                movable = True
                new_state = State(copy.deepcopy(self.board), get_opp_char(piece))
                new_state.board[y + 1][x - 1] = piece
                new_state.board[y][x] = '.'
                return movable, new_state

        return movable, None

    def attempt_move(self, coord):
        """
        Generate all possible moves for the given piece.

        Precondition: The index given must be a valid piece on the board, not an empty space. Also, the curr_turn
        must be the same as the colour of the piece that will be moved.
        :param coord: Coordinates of the piece on the board that we want to move
        :return: All possible moves that the piece given could carry out
        """
        possible_moves = []
        y = coord[0]
        x = coord[1]
        piece = self.board[y][x]  # piece can be 'b', 'r', 'B' or 'R'

        if piece in ['R', 'B']:  # King can move in any diagonal direction
            attempt_1 = self.check_move(coord, 'up_right')
            attempt_2 = self.check_move(coord, 'down_right')
            attempt_3 = self.check_move(coord, 'up_left')
            attempt_4 = self.check_move(coord, 'down_left')
            if attempt_1[0]:
                possible_moves.append(attempt_1[1])
            if attempt_2[0]:
                possible_moves.append(attempt_2[1])
            if attempt_3[0]:
                possible_moves.append(attempt_3[1])
            if attempt_4[0]:
                possible_moves.append(attempt_4[1])

        elif piece == 'r':  # 'r' can only move up-right or up-left
            attempt_1 = self.check_move(coord, 'up_left')
            attempt_2 = self.check_move(coord, 'up_right')
            if attempt_1[0]:
                possible_moves.append(attempt_1[1])
            if attempt_2[0]:
                possible_moves.append(attempt_2[1])

        elif piece == 'b':  # 'b' can only move down-right or down-left
            attempt_1 = self.check_move(coord, 'down_left')
            attempt_2 = self.check_move(coord, 'down_right')
            if attempt_1[0]:
                possible_moves.append(attempt_1[1])
            if attempt_2[0]:
                possible_moves.append(attempt_2[1])
        return possible_moves

    def generate_moves(self):
        """
        Generate all possible moves for the current state, that is all possible ways of moving every pieces one square
        diagonally to an adjacent unoccupied dark square. Normal pieces can move diagonally forward only;
        kings can move in any diagonal direction.
        :return: A list of possible new states after one move for every pieces that is on the player's turn
        """
        possible_moves = []

        for y, row in enumerate(self.board):
            for x, col in enumerate(row):
                if (self.curr_turn == ['b', 'B']) and (col in ['b', 'B']):  # black's turn and a black piece is found
                    possible_moves += self.attempt_move((y, x))
                elif (self.curr_turn == ['r', 'R']) and (col in ['r', 'R']):  # red's turn and red piece is found
                    possible_moves += self.attempt_move((y, x))
        # Check if any piece needs to be changed to king
        for s in possible_moves:
            s.check_king()
        return possible_moves

    def check_jump(self, coord, direction):
        """
        Check if the given direction of jump is possible. If yes, return the next state

        Precondition: the jump must not make the piece out of bound.
        :param coord: Coordinates of the piece on the board that we want to jump
        :param direction: Direction of the jump
        :return: A tuple: (True, new_state) or (False, None)
        """
        jumpable = False
        y = coord[0]
        x = coord[1]
        piece = self.board[y][x]  # piece can be 'b', 'r', 'B' or 'R'

        if direction == 'up_right':
            if y < 2 or x > 5: return jumpable, None  # check if the jump will be out of bound
            if (self.board[y - 1][x + 1] in get_opp_char(piece)) and (self.board[y - 2][x + 2] == '.'):
                jumpable = True
                # assume the next state has a jump -> don't change the player's turn
                new_state = State(copy.deepcopy(self.board), get_self_char(piece))
                new_state.board[y - 2][x + 2] = piece
                new_state.board[y - 1][x + 1] = '.'
                new_state.board[y][x] = '.'
                return jumpable, new_state
        elif direction == 'up_left':
            if y < 2 or x < 2: return jumpable, None  # check if the jump will be out of bound
            if (self.board[y - 1][x - 1] in get_opp_char(piece)) and (self.board[y - 2][x - 2] == '.'):
                jumpable = True
                new_state = State(copy.deepcopy(self.board), get_self_char(piece))
                new_state.board[y - 2][x - 2] = piece
                new_state.board[y - 1][x - 1] = '.'
                new_state.board[y][x] = '.'
                return jumpable, new_state
        elif direction == 'down_right':
            if y > 5 or x > 5: return jumpable, None  # check if the jump will be out of bound
            if (self.board[y + 1][x + 1] in get_opp_char(piece)) and (self.board[y + 2][x + 2] == '.'):
                jumpable = True
                new_state = State(copy.deepcopy(self.board), get_self_char(piece))
                new_state.board[y + 2][x + 2] = piece
                new_state.board[y + 1][x + 1] = '.'
                new_state.board[y][x] = '.'
                return jumpable, new_state
        elif direction == 'down_left':
            if y > 5 or x < 2: return jumpable, None  # check if the jump will be out of bound
            if (self.board[y + 1][x - 1] in get_opp_char(piece)) and (self.board[y + 2][x - 2] == '.'):
                jumpable = True
                new_state = State(copy.deepcopy(self.board), get_self_char(piece))
                new_state.board[y + 2][x - 2] = piece
                new_state.board[y + 1][x - 1] = '.'
                new_state.board[y][x] = '.'
                return jumpable, new_state

        return jumpable, None

    def attempt_jump(self, coord):
        """
        Generate all possible jumps for the given piece.

        Precondition: The index given must be a valid piece on the board, not an empty space. Also, the curr_turn
        must be the same as the colour of the piece that will be moved.
        :param coord: Coordinates of the piece on the board that we want to jump
        :return: All possible jumps that the piece given could carry out
        """
        possible_jumps = []
        new_coords = []
        y = coord[0]
        x = coord[1]
        piece = self.board[y][x]  # piece can be 'b', 'r', 'B' or 'R'

        if piece in ['R', 'B']:  # King can jump in any diagonal direction
            attempt_1 = self.check_jump(coord, 'up_right')
            attempt_2 = self.check_jump(coord, 'down_right')
            attempt_3 = self.check_jump(coord, 'up_left')
            attempt_4 = self.check_jump(coord, 'down_left')
            if attempt_1[0]:
                possible_jumps.append(attempt_1[1])
                new_coords.append((y - 2, x + 2))
            if attempt_2[0]:
                possible_jumps.append(attempt_2[1])
                new_coords.append((y + 2, x + 2))
            if attempt_3[0]:
                possible_jumps.append(attempt_3[1])
                new_coords.append((y - 2, x - 2))
            if attempt_4[0]:
                possible_jumps.append(attempt_4[1])
                new_coords.append((y + 2, x - 2))

        elif piece == 'r':  # 'r' can only jump up-right or up-left
            attempt_1 = self.check_jump(coord, 'up_left')
            attempt_2 = self.check_jump(coord, 'up_right')
            if attempt_1[0]:
                possible_jumps.append(attempt_1[1])
                new_coords.append((y - 2, x - 2))
            if attempt_2[0]:
                possible_jumps.append(attempt_2[1])
                new_coords.append((y - 2, x + 2))

        elif piece == 'b':  # 'b' can only jump down-right or down-left
            attempt_1 = self.check_jump(coord, 'down_left')
            attempt_2 = self.check_jump(coord, 'down_right')
            if attempt_1[0]:
                possible_jumps.append(attempt_1[1])
                new_coords.append((y + 2, x - 2))
            if attempt_2[0]:
                possible_jumps.append(attempt_2[1])
                new_coords.append((y + 2, x + 2))

        return possible_jumps, new_coords

    def find_more_jump(self, coord):
        """
        Given the coordinates of the piece we want to jump, find all of its possible jump sequences
        :param coord: Coordinates of the piece on the board that we want to jump
        :return: A list of states that are the last state of the jump sequence
        """
        y = coord[0]
        x = coord[1]

        jumps = self.attempt_jump((y, x))[0]  # A list of possible states after a jump
        coords = self.attempt_jump((y, x))[1]  # A list of coord of the piece after a jump

        jump_seq = []  # A list of possible states after a sequence of jumps
        if jumps:  # A jump exist for the given piece at the current state
            for j, c in zip(jumps, coords):
                if j.find_more_jump(c):  # continue finding if more jumps are available
                    jump_seq = jump_seq + j.find_more_jump(c)
                else:
                    jump_seq.append(j)  # no more jump is found
            return jump_seq
        else:  # Base case: the jump sequence ends
            return jump_seq

    def generate_jump(self):
        """
        Generate all possible jumps for the current state, that is "jumping over" the opponent's piece. Normal pieces
        can move diagonally forward only; kings can move in any diagonal direction.
        :return: A list of possible new states after one jump for every pieces that is on the player's turn
        """
        possible_jumps = []

        for y, row in enumerate(self.board):
            for x, col in enumerate(row):
                if (self.curr_turn == ['b', 'B']) and (col in ['b', 'B']):  # black's turn and a black piece is found
                    possible_jumps += self.find_more_jump((y, x))
                elif (self.curr_turn == ['r', 'R']) and (col in ['r', 'R']):  # red's turn and red piece is found
                    possible_jumps += self.find_more_jump((y, x))

        # Change the player's turn for the last state of the jump sequence
        for jump in possible_jumps:
            jump.curr_turn = get_opp_char(jump.curr_turn[0])
        # Check if any piece needs to be changed to king
        for s in possible_jumps:
            s.check_king()
        return possible_jumps

    def generate_successors(self):
        """
        :return: A list of successor states of the current state
        """
        successors = []
        jumps = self.generate_jump()
        if jumps:  # if you can jump, you must jump
            successors += jumps
        else:
            successors += self.generate_moves()
        return successors

    def is_terminal(self):
        """
        :return: True if the current state is a terminal state, i.e. either one player wins the game
        """
        is_red = False
        is_black = False
        for i in self.board:
            for j in i:
                if j in ['b', 'B']: is_black = True
                if j in ['r', 'R']: is_red = True
        is_terminal = not (is_red and is_black)
        winner = ''
        if is_terminal:
            if is_red: winner = 'r'
            elif is_black: winner = 'b'
        return is_terminal, winner

    def get_evaluation(self):
        """
        Assume red moves first and red is the max player
        Evaluation value are specified as follows:
            Pawn’s value: 5 + row number
            King’s value = 5 + 8(# of rows) + 2
        :return: The heuristic evaluation value of the given state, which is the estimated utility of the state
        """
        red_total = 0
        black_total = 0
        for y, row in enumerate(self.board):
            for p in row:
                if p == 'b':
                    black_total += 5 + y
                elif p == 'r':
                    red_total += 5 + (7 - y)
                elif p == 'B':
                    black_total += 15
                elif p == 'R':
                    red_total += 15
        return red_total - black_total

    def get_solution(self, output_file, running_time=0.0):
        """
        Given a goal state, backtrack through the parent state references until the initial state.
        Write the sequence of states into output_file
        :param running_time: running time of the function
        :param output_file: File that the result is stored
        :type output_file: str
        :return: None
        """
        path = []
        f = open(output_file, "w")
        while self.parent:
            path.append(self)
            self = self.parent
        path.append(self)
        path = path[::-1]
        for states in path:
            for i, line in enumerate(states.board):
                for ch in line:
                    f.write(ch)
                f.write("\n")
            f.write("\n")
        # f.write(f"{len(path) - 1} moves\n")
        # f.write(f"{running_time} seconds")
        f.close()

    def check_king(self):
        """
        Check if the current state has any piece that needs to be crowned as king
        :return: None
        """
        for i, piece in enumerate(self.board[0]):
            if piece == 'r':
                self.board[0][i] = 'R'
        for i, piece in enumerate(self.board[7]):
            if piece == 'b':
                self.board[0][i] = 'B'


class Solver:
    def __init__(self, init_state: State):
        self.curr_state = init_state
        self.cache = {}  # A dictionary with state as key, heuristic as value

    def play_game(self):
        """
        :return: Given the initial state, return the last state where red wins
        """
        if self.curr_state.is_terminal()[0]:  # Assuming all game must terminate and red must win
            return self.curr_state
        else:
            self.curr_state = self.alpha_beta_search(self.curr_state)
            return self.play_game()

    def alpha_beta_search(self, state, depth_limit=12) -> State:
        """
        Assume red moves first and red is the max player
        :return: The best next move a player could make
        """
        best_successor = None
        if state.curr_turn == ['r', 'R']:
            best_successor, _ = self.max_value(state, -float('inf'), float('inf'), depth_limit)
        elif state.curr_turn == ['b', 'B']:
            best_successor, _ = self.min_value(state, -float('inf'), float('inf'), depth_limit)

        if best_successor is not None:
            best_successor.parent = self.curr_state
        return best_successor

    def max_value(self, state, alpha, beta, depth_limit):
        """
        Minimax function for finding max node with alpha-beta pruning
        :return: The best successor for the max player and its utility value
        """
        best_successor = None
        # Under either one of the three cases below, the search is stopped
        if state.is_terminal()[0]:
            return state, get_utility(depth_limit, state.is_terminal()[1])
        elif str(state) in self.cache:
            return self.cache[str(state)][0], self.cache[str(state)][1]
        elif depth_limit == 0:
            return state, state.get_evaluation()

        value = -float('inf')
        successors = state.generate_successors()
        # sort the list of states in successors in descending order
        successors.sort(key=State.get_evaluation, reverse=True)
        for s in successors:
            _, temp_v = self.min_value(s, alpha, beta, depth_limit - 1)
            # update v and best_successor if the new utility is higher
            if value < temp_v:
                value = temp_v
                best_successor = s
            # alpha-beta pruning
            if value > beta:
                return best_successor, value
            alpha = max(alpha, value)
        # cache the state with its corresponding best successor and heuristic value
        self.cache[str(state)] = best_successor, value
        return best_successor, value

    def min_value(self, state, alpha, beta, depth_limit):
        """
        Minimax function for finding min node with alpha-beta pruning
        :return: The best successor for the min player and its utility value
        """
        best_successor = None
        # Under either one of the three cases below, the search is stopped
        if state.is_terminal()[0]:
            return state, get_utility(depth_limit,  state.is_terminal()[1])
        elif str(state) in self.cache:
            return self.cache[str(state)][0], self.cache[str(state)][1]
        elif depth_limit == 0:
            return state, state.get_evaluation()

        value = float('inf')
        successors = state.generate_successors()
        # sort the list of states in successors in ascending order
        successors.sort(key=State.get_evaluation, reverse=False)
        for s in successors:
            _, temp_v = self.max_value(s, alpha, beta, depth_limit - 1)
            # update v and best_successor if the new utility is higher
            if value > temp_v:
                value = temp_v
                best_successor = s
            # alpha-beta pruning
            if value < alpha:
                return best_successor, value
            beta = min(beta, value)
        # cache the state with its corresponding best successor and heuristic value
        self.cache[str(state)] = best_successor, value
        return best_successor, value


def get_utility(depth, winner):
    """
    The utility value is intentionally set way higher than the eval value, since we only evaluate the utility value
    on terminal states, Thus, setting a higher value could prioritize the search towards the terminal state.
    :return: The utility value of the state
    """
    utility = 0
    if winner == 'r':
        utility = 1000
    if winner == 'b':
        utility = -1000
    return utility * depth


def get_opp_char(player):
    if player in ['b', 'B']:
        return ['r', 'R']
    else:
        return ['b', 'B']


def get_self_char(player):
    if player in ['b', 'B']:
        return ['b', 'B']
    else:
        return ['r', 'R']


def read_from_file(filename):
    f = open(filename)
    lines = f.readlines()
    board = [[str(x) for x in l.rstrip()] for l in lines]
    f.close()

    return board


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzles."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    args = parser.parse_args()

    initial_board = read_from_file(args.inputfile)
    state = State(initial_board, ['r', 'R'])
    solver = Solver(state)
    final_state = solver.play_game()
    final_state.get_solution(args.outputfile)
    sys.stdout = sys.__stdout__

    # initial_board = read_from_file("checkers1.txt")
    # init_state = State(initial_board, ['r', 'R'])
    #
    # solver = Solver(init_state)
    # start_time = time.time()
    # final_state = solver.play_game()
    # end_time = time.time()
    # running_time = end_time - start_time
    # final_state.get_solution("test_sol_output.txt", running_time)


