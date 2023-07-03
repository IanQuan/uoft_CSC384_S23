from heapq import heappush, heappop
import argparse

# ====================================================================================

char_goal = '1'
char_single = '2'


class Piece:
    """
    This represents a piece on the Hua Rong Dao puzzle.
    """

    def __init__(self, is_goal, is_single, coord_x, coord_y, orientation):
        """
        :param is_goal: True if the piece is the goal piece and False otherwise.
        :type is_goal: bool
        :param is_single: True if this piece is a 1x1 piece and False otherwise.
        :type is_single: bool
        :param coord_x: The x coordinate of the top left corner of the piece.
        :type coord_x: int
        :param coord_y: The y coordinate of the top left corner of the piece.
        :type coord_y: int
        :param orientation: The orientation of the piece (one of 'h' or 'v')
            if the piece is a 1x2 piece. Otherwise, this is None
        :type orientation: str
        """

        self.is_goal = is_goal
        self.is_single = is_single
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.orientation = orientation

    def __repr__(self):
        return '{} {} {} {} {}'.format(self.is_goal, self.is_single,
                                       self.coord_x, self.coord_y, self.orientation)


class Board:
    """
    Board class for setting up the playing board.
    """

    def __init__(self, pieces):
        """
        :param pieces: The list of Pieces
        :type pieces: List[Piece]
        """

        self.width = 4
        self.height = 5

        self.pieces = pieces

        # self.grid is a 2-d (size * size) array automatically generated
        # using the information on the pieces when a board is being created.
        # A grid contains the symbol for representing the pieces on the board.
        self.grid = []
        self.__construct_grid()

    def __str__(self):
        """
        :return: str representation of the board
        """

        grid_str = ""
        for row in self.grid:
            for ch in row:
                grid_str += str(ch) + " "
            grid_str += "\n"
        return grid_str

    def __construct_grid(self):
        """
        Called in __init__ to set up a 2-d grid based on the piece location information.

        """

        for i in range(self.height):
            line = []
            for j in range(self.width):
                line.append('.')
            self.grid.append(line)

        for piece in self.pieces:
            if piece.is_goal:
                self.grid[piece.coord_y][piece.coord_x] = char_goal
                self.grid[piece.coord_y][piece.coord_x + 1] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x + 1] = char_goal
            elif piece.is_single:
                self.grid[piece.coord_y][piece.coord_x] = char_single
            else:
                if piece.orientation == 'h':
                    self.grid[piece.coord_y][piece.coord_x] = '<'
                    self.grid[piece.coord_y][piece.coord_x + 1] = '>'
                elif piece.orientation == 'v':
                    self.grid[piece.coord_y][piece.coord_x] = '^'
                    self.grid[piece.coord_y + 1][piece.coord_x] = 'v'

    def display(self):
        """
        Print out the current board.

        """

        for i, line in enumerate(self.grid):
            for ch in line:
                print(ch, end='')
            print()

    def find_empty(self):
        """
        :return: The coordinates of the empty spaces in a list [(x1, y1), (x2, y2)]
        """

        empty_spaces = []
        for y_cord, row in enumerate(self.grid):
            for x_cord, ch in enumerate(row):
                if ch == '.':
                    empty_spaces.append((x_cord, y_cord))
        return empty_spaces

    def move(self, index,  move_dir):
        """
        Move a piece from a given direction
        Pre-condition: given the piece and the direction, the move must be
        valid (moving to an empty space without overlapping with other pieces)

        :param index: Index of the piece in pieces that will be moved
        :type index: int
        :param move_dir: Direction of the movement
        :type move_dir: str
        :return: The new board after a piece is moved
        """

        new_piece = Piece(False, False, 0, 0, None)
        new_pieces = []
        old_piece = self.pieces[index]
        inserted = False

        # create a new piece with the new coordinates
        if move_dir == 'up':
            new_piece = Piece(old_piece.is_goal, old_piece.is_single, old_piece.coord_x,
                              old_piece.coord_y - 1, old_piece.orientation)
        elif move_dir == 'down':
            new_piece = Piece(old_piece.is_goal, old_piece.is_single, old_piece.coord_x,
                              old_piece.coord_y + 1, old_piece.orientation)
        elif move_dir == 'right':
            new_piece = Piece(old_piece.is_goal, old_piece.is_single, old_piece.coord_x + 1,
                              old_piece.coord_y, old_piece.orientation)
        elif move_dir == 'left':
            new_piece = Piece(old_piece.is_goal, old_piece.is_single, old_piece.coord_x - 1,
                              old_piece.coord_y, old_piece.orientation)

        # create a new list new_pieces without the old_piece and insert the new_piece accordingly
        for p in self.pieces:
            if repr(p) == repr(old_piece):
                continue
            if not inserted:
                if(new_piece.coord_y < p.coord_y) or (new_piece.coord_y == p.coord_y and new_piece.coord_x < p.coord_x):
                    new_pieces.append(new_piece)
                    inserted = True
            new_pieces.append(p)

        if not inserted:
            new_pieces.append(new_piece)

        new_board = Board(new_pieces)
        return new_board

    def manhattan(self):
        """
        :return: The vertical distance of the 2x2 goal piece (top-left coord) from the goal position (1 3)
        """

        for p in self.pieces:
            if p.is_goal:
                return abs(1 - p.coord_x) + abs(3 - p.coord_y)


class State:
    """
    State class wrapping a Board with some extra current state information.
    Note that State and Board are different. Board has the locations of the pieces. 
    State has a Board and some extra information that is relevant to the search: 
    heuristic function, f value, current depth and parent.
    """

    def __init__(self, board, f, depth, parent=None):
        """
        :param board: The board of the state.
        :type board: Board
        :param f: The f value of current state.
        :type f: int
        :param depth: The depth of current state in the search tree.
        :type depth: int
        :param parent: The parent of current state.
        :type parent: Optional[State]
        """

        self.board = board
        self.f = f
        self.depth = depth
        self.parent = parent
        self.id = hash(board)  # The id for breaking ties.

    def get_solution(self, output_file):
        """
        Given a goal state, backtrack through the parent state references until the initial state.
        Write the sequence of states into output_file

        :param output_file: File that the result is stored
        :type output_file: str
        :return: None
        """

        path = []
        f = open(output_file, "a")
        while self.parent:
            path.append(self)
            self = self.parent
        path.append(self)
        path = path[::-1]
        for states in path:
            for i, line in enumerate(states.board.grid):
                for ch in line:
                    f.write(ch)     # ends with a new line
                f.write("\n")
            f.write("\n")
        f.close()


class Solver:
    """
    Solver class contains all the required information for running the search algorithm on the problem
    """
    def __init__(self, init_state):
        """
        :param init_state: Initial state of the algo
        :type init_state: State
        """

        self.init_state = init_state
        self.curr_state = init_state    # The current state
        self.frontier = []              # The list of successor states that is successfully added
        self.explored = set()           # The set of string representation of explored states

    def check_goal(self):
        """
        :return: True if curr_state is the goal state
        """

        curr_board = self.curr_state.board
        for p in curr_board.pieces:
            if repr(p) == 'True False 1 3 None':
                return True
        return False

    def generate_state(self, index, move_dir, algo):
        """
        Given the piece that will be moved and its direction, create the next state according to the movement
        for the curr_state

        :param index: Index of the piece in the list pieces of  board
        :type index: int
        :param move_dir: Direction of the piece movement
        :type move_dir: str
        :param algo: "DFS" or "A*"
        :type algo: str
        :return: One successor state of the curr_state
        """

        curr_depth = self.curr_state.depth
        curr_board = self.curr_state.board

        new_board = curr_board.move(index, move_dir)

        if algo == "DFS":
            new_state = State(new_board, 0, curr_depth + 1, self.curr_state)
            return new_state
        elif algo == "A*":
            f = new_board.manhattan() + curr_depth
            new_state = State(new_board, f, curr_depth + 1, self.curr_state)
            return new_state

    def generate_all_successors(self, algo):
        """
        Given the curr_state, find all the possible ways to move a piece and move it.
        In other words, find all possible successors

        :param algo: "DFS" or "A*"
        :type algo: str
        :return: A list of its successor states
        """

        successors = []
        curr_board = self.curr_state.board

        # get the coordinates of the two empty spaces e1, e2
        e1_x = curr_board.find_empty()[0][0]
        e1_y = curr_board.find_empty()[0][1]
        e2_x = curr_board.find_empty()[1][0]
        e2_y = curr_board.find_empty()[1][1]
        # print(f'({e1_x}, {e1_y}), ({e2_x}, {e2_y})')

        # iterates through all pieces and search for possible movements -> add all the possible successors into a list
        for i, p in enumerate(curr_board.pieces):
            # goal state
            if p.is_goal:
                # move up?
                if (p.coord_y == e1_y + 1) and (p.coord_x == e1_x) and (e2_x == e1_x + 1) and (e2_y == e1_y):
                    new_state = self.generate_state(i, 'up', algo)
                    successors.append(new_state)
                # move down?
                if (p.coord_y == e1_y - 2) and (p.coord_x == e1_x) and (e2_x == e1_x + 1) and (e2_y == e1_y):
                    new_state = self.generate_state(i, 'down', algo)
                    successors.append(new_state)
                # move right?
                if (p.coord_y == e1_y) and (p.coord_x == e1_x - 2) and (e2_x == e1_x) and (e2_y == e1_y + 1):
                    new_state = self.generate_state(i, 'right', algo)
                    successors.append(new_state)
                # move left?
                if (p.coord_y == e1_y) and (p.coord_x == e1_x + 1) and (e2_x == e1_x) and (e2_y == e1_y + 1):
                    new_state = self.generate_state(i, 'left', algo)
                    successors.append(new_state)
            # 1x1 single state
            elif p.is_single:
                # move up?
                if (p.coord_y == e1_y + 1 and p.coord_x == e1_x) or (p.coord_y == e2_y + 1 and p.coord_x == e2_x):
                    new_state = self.generate_state(i, 'up', algo)
                    successors.append(new_state)
                # move down?
                if (p.coord_y == e1_y - 1 and p.coord_x == e1_x) or (p.coord_y == e2_y - 1 and p.coord_x == e2_x):
                    new_state = self.generate_state(i, 'down', algo)
                    successors.append(new_state)
                # move right?
                if (p.coord_y == e1_y and p.coord_x == e1_x - 1) or (p.coord_y == e2_y and p.coord_x == e2_x - 1):
                    new_state = self.generate_state(i, 'right', algo)
                    successors.append(new_state)
                # move left?
                if (p.coord_y == e1_y and p.coord_x == e1_x + 1) or (p.coord_y == e2_y and p.coord_x == e2_x + 1):
                    new_state = self.generate_state(i, 'left', algo)
                    successors.append(new_state)
            else:
                # 2x1 horizontal state
                if p.orientation == 'h':
                    # move up?
                    if (p.coord_y == e1_y + 1) and (p.coord_x == e1_x) and (e1_x + 1 == e2_x) and (e1_y == e2_y):
                        new_state = self.generate_state(i, 'up', algo)
                        successors.append(new_state)
                    # move down?
                    if (p.coord_y == e1_y - 1) and (p.coord_x == e1_x) and (e1_x + 1 == e2_x) and (e1_y == e2_y):
                        new_state = self.generate_state(i, 'down', algo)
                        successors.append(new_state)
                    # move right?
                    if (p.coord_y == e1_y and p.coord_x == e1_x - 2) or (p.coord_y == e2_y and p.coord_x == e2_x - 2):
                        new_state = self.generate_state(i, 'right', algo)
                        successors.append(new_state)
                    # move left?
                    if (p.coord_y == e1_y and p.coord_x == e1_x + 1) or (p.coord_y == e2_y and p.coord_x == e2_x + 1):
                        new_state = self.generate_state(i, 'left', algo)
                        successors.append(new_state)
                # 1x2 vertical state
                elif p.orientation == 'v':
                    # move up?
                    if (p.coord_y == e1_y + 1 and p.coord_x == e1_x) or (p.coord_y == e2_y + 1 and p.coord_x == e2_x):
                        new_state = self.generate_state(i, 'up', algo)
                        successors.append(new_state)
                    # move down?
                    if (p.coord_y == e1_y - 2 and p.coord_x == e1_x) or (p.coord_y == e2_y - 2 and p.coord_x == e2_x):
                        new_state = self.generate_state(i, 'down', algo)
                        successors.append(new_state)
                    # move right?
                    if (p.coord_y == e1_y) and (p.coord_x == e1_x - 1) and (e1_x == e2_x) and (e1_y == e2_y - 1):
                        new_state = self.generate_state(i, 'right', algo)
                        successors.append(new_state)
                    # move left?
                    if (p.coord_y == e1_y) and (p.coord_x == e1_x + 1) and (e1_x == e2_x) and (e1_y == e2_y - 1):
                        new_state = self.generate_state(i, 'left', algo)
                        successors.append(new_state)
        return successors

    def dfs(self, output_file):
        """
        Run depth-first search with the given init_state
        :param output_file: File with a sequence of states, optimal solution found by DFS
        :type output_file: str
        """

        self.frontier.append(self.init_state)

        while len(self.frontier) > 0:
            # pop the newest state from teh frontier
            self.curr_state = self.frontier.pop(-1)
            c_s = self.curr_state
            # c_s.board.display()
            # print("\n")
            # check if the current state been explored
            if c_s.board.__str__() not in self.explored:
                # add current state to explored set
                self.explored.add(c_s.board.__str__())
                # terminate if the goal state is found
                if self.check_goal():
                    c_s.get_solution(output_file)
                    return
                # find the successors and add them to the frontier if not yet explored
                for suc in self.generate_all_successors('DFS'):
                    if suc.board.__str__() not in self.explored:
                        self.frontier.append(suc)
        print("No solution found.")

    def a_star_manhattan(self, output_file):
        """
        Run A* search with Manhattan heuristics to find the optimal solution
            - frontier is a min-heap with f(State) as the key and State as the value
            - f(State) = State.cost + State.board.manhattan
            - Tie-breaking rule: select the state with lower State.id value
        :param output_file: File with a sequence of states, optimal solution found by A* search
        :type output_file: str
        """
        heappush(self.frontier, (self.curr_state.f, self.curr_state.id, self.curr_state))

        while len(self.frontier) > 0:
            # pop the state with the smallest f(State) from the frontier
            self.curr_state = heappop(self.frontier)[2]
            c_s = self.curr_state
            # c_s.board.display()
            # print("\n")
            # check if the current state been explored
            if c_s.board.__str__() not in self.explored:
                # add current state to explored set
                self.explored.add(c_s.board.__str__())
                # terminate if the goal state is found
                if self.check_goal():
                    c_s.get_solution(output_file)
                    return
                # find the successors and add them to the frontier if not yet explored
                for suc in self.generate_all_successors('A*'):
                    if suc.board.__str__() not in self.explored:
                        heappush(self.frontier, (suc.f, suc.id, suc))
        print("No solution found.")


def read_from_file(filename):
    """
    Load initial board from a given file.

    :param filename: The name of the given file.
    :type filename: str

    :return: A loaded board
    :rtype: Board
    """

    puzzle_file = open(filename, "r")

    line_index = 0
    pieces = []
    g_found = False

    for line in puzzle_file:

        for x, ch in enumerate(line):

            if ch == '^':  # found vertical piece
                pieces.append(Piece(False, False, x, line_index, 'v'))
            elif ch == '<':  # found horizontal piece
                pieces.append(Piece(False, False, x, line_index, 'h'))
            elif ch == char_single:
                pieces.append(Piece(False, True, x, line_index, None))
            elif ch == char_goal:
                if g_found == False:
                    pieces.append(Piece(True, False, x, line_index, None))
                    g_found = True
        line_index += 1

    puzzle_file.close()

    board = Board(pieces)
    
    return board


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzle."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=['astar', 'dfs'],
        help="The searching algorithm."
    )
    args = parser.parse_args()

    # read the board from the file
    board = read_from_file(args.inputfile)

    state = State(board, 0, 0, None)
    solvers = Solver(state)
    if args.algo == "dfs":
        solvers.dfs(args.outputfile)
    elif args.algo == 'astar':
        solvers.a_star_manhattan(args.outputfile)

    # board_1 = read_from_file('testhrd_easy1.txt')
    # board_1.display()
    # start_state_1 = State(board_1, 0, 0)
    # solver_1 = Solver(start_state_1)
    # solver_1.a_star_manhattan("testhrd_easy_sol.txt")
    # print("----------------------------------------------------------------------------------------------------")
    # board_2 = read_from_file('testhrd_med1.txt')
    # board_2.display()
    # start_state_2 = State(board_2, 0, 0)
    # solver_2 = Solver(start_state_2)
    # solver_2.a_star_manhattan('testhard_med1_sol.txt')
    # print("----------------------------------------------------------------------------------------------------")
    # board_3 = read_from_file('testhrd_hard1.txt')
    # board_3.display()
    # start_state_3 = State(board_3, 0, 0)
    # solver_3 = Solver(start_state_3)
    # solver_3.a_star_manhattan('test_hard1_sol.txt')

