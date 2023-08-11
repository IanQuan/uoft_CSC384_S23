import argparse
import math
import sys
import copy
from csp import Variable, CSP
from constraints import NValuesConstraint, TableConstraint


def read_input(file):
    """
    :param file: File name of the input
    Read the input fil.,
    :return: row constraints, column constraints, ship constraints as a list of int; board as a 2D array of char
    """

    f = open(file, 'r')
    b = f.read().split()
    size = len(b[0])

    row_cons = [int(i) for i in b[0]]  # read row constraints as a list of int
    col_cons = [int(i) for i in b[1]]  # read column constraints a list of int

    # Store the ship variables into self.ships
    b[2] += ('0' if len(b[2]) == 3 else '')  # Ensure the number of type of ships is 4
    ship_cons = []
    for i in b[2]:  # b[2] describes the number of each type of ship.
        ship_cons.append(int(i))  # read ship constraints as a list of int

    # Store only the board section of the input to board
    board = [[str(0) for _ in range(size)] for _ in range(size)]
    for row in range(3, size + 3):
        for i, col in enumerate(b[row]):
            board[row - 3][i] = col

    return row_cons, col_cons, ship_cons, board


def create_variable(board):
    """
    :param board: Initial board of the input
    Create variable for each cell and set their respective domains as a list of possible symbols
    ['S', '.', '<', '>', '^', 'v', 'M']
    :return:A list of Variable and a dictionary with key = var_name, value = Variable
    """

    size = len(board[0])
    var_list = []  # list of Variable Object
    var_dict = {}  # dictionary with key = var_name, value = Variable

    for row in range(size):
        for col in range(size):
            var_name = str(row * size + col)

            if board[row][col] == '0':  # if the cell has '0' -> domain = all possible symbol
                var = Variable(var_name, ['S', '.', '<', '>', '^', 'v', 'M'])
            else:  # if the cell has a symbol already -> domain = its own symbol
                var = Variable(var_name, [board[row][col]])

            var_list.append(var)
            var_dict[var_name] = var

    return var_list, var_dict


def create_constraints(board, row_con, col_con, var_dict):
    constraints = []
    size = len(board[0])
    for y in range(size):

        row_variables = []
        col_variables = []
        for x in range(size):
            row_variables.append(var_dict[str(y * size + x)])
            col_variables.append(var_dict[str(y + x * size)])
            # Add constraints for neighbour cells
            constraints += check_around(board, y, x, var_dict)

        if y == 0:  # first row and column
            row_values = ['S', '<', '>', '^', 'M']
            col_values = ['S', '<', '^', 'v', 'M']
        elif y == size - 1:  # last row and column
            row_values = ['S', '<', '>', 'v', 'M']
            col_values = ['S', '>', '^', 'v', 'M']
        else:
            row_values = ['S', '<', '>', '^', 'v', 'M']
            col_values = ['S', '<', '>', '^', 'v', 'M']
        # Add row constraints and col constraints
        constraints.append(NValuesConstraint('row_' + str(y), row_variables, row_values, row_con[y], row_con[y]))
        constraints.append(NValuesConstraint('col_' + str(y), col_variables, col_values, col_con[y], col_con[y]))

    return constraints


def check_helper(y, x, size, var_dict):
    """
    Helper function for check_around. For all 8 directions around a cell, instantiate its corresponding tuple
    (name (str), scope of cells (list)) only when necessary according to the coordinates of the cell given .
    :return: A list of tuple [(name (str), scope of cells (list))]
    """

    cell_name = str(y * size + x)
    left, right, bottom_right, bottom_left, top_right, top_left, top, bottom = (), (), (), (), (), (), (), ()

    if x < size - 1:
        right = ('cell_' + str(y) + str(x) + '_to_cell_' + str(y) + str(x + 1),
                 [var_dict[cell_name], var_dict[str(y * size + x + 1)]])
        if y < size - 1:
            bottom_right = ('cell_' + str(y) + str(x) + '_to_cell_' + str(y + 1) + str(x + 1),
                            [var_dict[cell_name], var_dict[str((y + 1) * size + x + 1)]])
        if y > 0:
            top_right = ('cell_' + str(y) + str(x) + '_to_cell_' + str(y - 1) + str(x + 1),
                         [var_dict[cell_name], var_dict[str((y - 1) * size + x + 1)]])
    if x > 0:
        left = ('cell_' + str(y) + str(x) + '_to_cell_' + str(y) + str(x - 1),
                [var_dict[cell_name], var_dict[str(y * size + x - 1)]])
        if y < size - 1:
            bottom_left = ('cell_' + str(y) + str(x) + '_to_cell_' + str(y + 1) + str(x - 1),
                           [var_dict[cell_name], var_dict[str((y + 1) * size + x - 1)]])
        if y > 0:
            top_left = ('cell_' + str(y) + str(x) + '_to_cell_' + str(y - 1) + str(x - 1),
                        [var_dict[cell_name], var_dict[str((y - 1) * size + x - 1)]])
    if y > 0:
        top = ('cell_' + str(y) + str(x) + '_to_cell_' + str(y - 1) + str(x),
               [var_dict[cell_name], var_dict[str((y - 1) * size + x)]])
    if y < size - 1:
        bottom = ('cell_' + str(y) + str(x) + '_to_cell_' + str(y + 1) + str(x),
                  [var_dict[cell_name], var_dict[str((y + 1) * size + x)]])

    return left, right, bottom_right, bottom_left, top_right, top_left, top, bottom


def check_around(board, y, x, var_dict):
    """
    Given the coordinates of the cell, add constraints with appropriate neighbour cell, where the constraints
    are all the possible combination of symbol combinations with that specific neighbour cell.
    :param board: Board from the input file
    :param y: y-coordinates of the current cell
    :param x: x-coordinates of the current cell
    :param var_dict: Dictionary with key = var_name, value = Variable
    :return: A list of TableConstraint for the cell given 
    """
    size = len(board[0])
    constraint = []
    left, right, bottom_right, bottom_left, top_right, top_left, top, bottom = check_helper(y, x, size, var_dict)

    # Create constraints according to the position of cell
    if y == 0 and x == 0:  # cell is in the top-left corner
        constraint.append(TableConstraint(right[0], right[1],
                                          [['S', '.'], ['.', '^'], ['.', '<'], ['.', 'S'], ['.', '.'], ['<', '>'],
                                           ['<', 'M'], ['^', '.']]))

        constraint.append(TableConstraint(bottom_right[0], bottom_right[1],
                                          [['S', '.'], ['.', '^'], ['.', 'v'], ['.', '<'], ['.', 'S'], ['.', '.'],
                                           ['.', 'M'], ['.', '>'], ['<', '.'], ['^', '.']]))

        constraint.append(TableConstraint(bottom[0], bottom[1],
                                          [['S', '.'], ['.', '^'], ['.', '<'], ['.', 'S'], ['.', '.'], ['<', '.'],
                                           ['^', 'v'], ['^', 'M']]))

    elif y == 0 and x == size - 1:  # cell is in the top-right corner
        constraint.append(TableConstraint(left[0], left[1],
                                          [['S', '.'], ['.', '^'], ['.', '>'], ['.', 'S'], ['.', '.'], ['>', '<'],
                                           ['>', 'M'], ['^', '.']]))

        constraint.append(TableConstraint(bottom_left[0], bottom_left[1],
                                          [['S', '.'], ['.', '^'], ['.', 'v'], ['.', '<'], ['.', 'S'], ['.', '.'],
                                           ['.', 'M'], ['.', '>'], ['>', '.'], ['^', '.']]))

        constraint.append(TableConstraint(bottom[0], bottom[1],
                                          [['S', '.'], ['.', '^'], ['.', '>'], ['.', 'S'], ['.', '.'], ['>', '.'],
                                           ['^', 'v'], ['^', 'M']]))

    elif y == size - 1 and x == 0:  # cell is in the bottom-left corner
        constraint.append(TableConstraint(right[0], right[1],
                                          [['S', '.'], ['.', 'v'], ['.', '<'], ['.', 'S'], ['.', '.'], ['<', '>'],
                                           ['<', 'M'], ['v', '.']]))

        constraint.append(TableConstraint(top_right[0], top_right[1],
                                          [['S', '.'], ['.', '^'], ['.', 'v'], ['.', '<'], ['.', 'S'], ['.', '.'],
                                           ['.', 'M'], ['.', '>'], ['<', '.'], ['v', '.']]))

        constraint.append(TableConstraint(top[0], top[1],
                                          [['S', '.'], ['.', 'v'], ['.', '<'], ['.', 'S'], ['.', '.'], ['<', '.'],
                                           ['v', '^'], ['v', 'M']]))

    elif y == size - 1 and x == size - 1:  # cell is in the bottom-right corner
        constraint.append(TableConstraint(left[0], left[1],
                                          [['S', '.'], ['.', 'v'], ['.', '>'], ['.', 'S'], ['.', '.'], ['>', '<'],
                                           ['>', 'M'], ['v', '.']]))

        constraint.append(TableConstraint(top_left[0], top_left[1],
                                          [['S', '.'], ['.', '^'], ['.', 'v'], ['.', '<'], ['.', 'S'], ['.', '.'],
                                           ['.', 'M'], ['.', '>'], ['>', '.'], ['v', '.']]))

        constraint.append(TableConstraint(top[0], top[1],
                                          [['S', '.'], ['.', 'v'], ['.', '>'], ['.', 'S'], ['.', '.'], ['>', '.'],
                                           ['v', '^'], ['v', 'M']]))

    elif y == 0:  # cell is on the top row
        constraint.append(TableConstraint(bottom[0], bottom[1],
                                          [['S', '.'], ['.', '^'], ['.', '>'], ['.', '<'], ['.', 'S'], ['.', '.'],
                                           ['.', 'M'], ['>', '.'], ['^', 'v'], ['^', 'M'], ['<', '.'], ['M', '.']]))

        constraint.append(TableConstraint(left[0], left[1],
                                          [['S', '.'], ['.', '^'], ['.', '>'], ['.', 'S'], ['.', '.'], ['>', '<'],
                                           ['>', 'M'], ['^', '.'], ['<', '.'], ['M', '<'], ['M', 'M']]))

        constraint.append(TableConstraint(right[0], right[1],
                                          [['S', '.'], ['.', '^'], ['.', '<'], ['.', 'S'], ['.', '.'], ['<', '>'],
                                           ['<', 'M'], ['^', '.'], ['>', '.'], ['M', '>'], ['M', 'M']]))

        constraint.append(TableConstraint(bottom_left[0], bottom_left[1],
                                          [['S', '.'], ['.', '^'], ['.', 'v'], ['.', '<'], ['.', 'S'], ['.', '.'],
                                           ['.', 'M'], ['.', '>'], ['>', '.'], ['^', '.'], ['<', '.'], ['M', '.']]))

        constraint.append(TableConstraint(bottom_right[0], bottom_right[1],
                                          [['S', '.'], ['.', '^'], ['.', 'v'], ['.', '<'], ['.', 'S'], ['.', '.'],
                                           ['.', 'M'], ['.', '>'], ['>', '.'], ['^', '.'], ['<', '.'], ['M', '.']]))

    elif x == 0:  # cell is on the left-most column
        constraint.append(TableConstraint(bottom[0], bottom[1],
                                          [['S', '.'], ['.', '^'], ['.', '<'], ['.', 'S'], ['.', '.'], ['^', 'v'],
                                           ['^', 'M'], ['<', '.'], ['M', 'M'], ['M', 'v'], ['v', '.']]))

        constraint.append(TableConstraint(right[0], right[1],
                                          [['S', '.'], ['.', 'v'], ['.', '<'], ['.', 'S'], ['.', '.'], ['<', '>'],
                                           ['<', 'M'], ['v', '.'], ['^', '.'], ['.', '^'], ['M', '.'], ['.', 'M']]))

        constraint.append(TableConstraint(top_right[0], top_right[1],
                                          [['S', '.'], ['.', 'v'], ['.', '^'], ['.', '<'], ['.', 'S'], ['.', '.'],
                                           ['.', 'M'], ['.', '>'], ['v', '.'], ['^', '.'], ['<', '.'], ['M', '.']]))

        constraint.append(TableConstraint(bottom_right[0], bottom_right[1],
                                          [['S', '.'], ['.', 'v'], ['.', '^'], ['.', '<'], ['.', 'S'], ['.', '.'],
                                           ['.', 'M'], ['.', '>'], ['v', '.'], ['^', '.'], ['<', '.'], ['M', '.']]))

        constraint.append(TableConstraint(top[0], top[1],
                                          [['S', '.'], ['.', 'v'], ['.', '<'], ['.', 'S'], ['.', '.'], ['v', '^'],
                                           ['v', 'M'], ['<', '.'], ['M', 'M'], ['M', '^'], ['^', '.']]))

    elif y == size - 1:  # cell is on the bottom row
        constraint.append(TableConstraint(top[0], top[1],
                                          [['S', '.'], ['.', 'v'], ['.', '>'], ['.', '<'], ['.', 'S'], ['.', '.'],
                                           ['.', 'M'], ['>', '.'], ['<', '.'], ['v', '^'], ['v', 'M'], ['M', '.']]))

        constraint.append(TableConstraint(left[0], left[1],
                                          [['S', '.'], ['.', 'v'], ['.', '>'], ['.', 'S'], ['.', '.'], ['>', '<'],
                                           ['>', 'M'], ['v', '.'], ['<', '.'], ['M', '<'], ['M', 'M']]))

        constraint.append(TableConstraint(right[0], right[1],
                                          [['S', '.'], ['.', 'v'], ['.', '<'], ['.', 'S'], ['.', '.'], ['<', '>'],
                                           ['<', 'M'], ['v', '.'], ['>', '.'], ['M', '>'], ['M', 'M']]))

        constraint.append(TableConstraint(top_left[0], top_left[1],
                                          [['S', '.'], ['.', 'v'], ['.', '^'], ['.', '<'], ['.', 'S'], ['.', '.'],
                                           ['.', 'M'], ['.', '>'], ['>', '.'], ['v', '.'], ['<', '.'], ['M', '.']]))

        constraint.append(TableConstraint(top_right[0], top_right[1],
                                          [['S', '.'], ['.', 'v'], ['.', '^'], ['.', '<'], ['.', 'S'], ['.', '.'],
                                           ['.', 'M'], ['.', '>'], ['>', '.'], ['v', '.'], ['<', '.'], ['M', '.']]))

    elif x == size - 1:  # cell is on the right-most column
        constraint.append(TableConstraint(left[0], left[1],
                                          [['S', '.'], ['.', 'v'], ['.', '>'], ['.', 'S'], ['.', '.'], ['>', '<'],
                                           ['>', 'M'], ['v', '.'], ['^', '.'], ['.', '^'], ['M', '.'], ['.', 'M']]))

        constraint.append(TableConstraint(top_left[0], top_left[1],
                                          [['S', '.'], ['.', 'v'], ['.', '^'], ['.', '<'], ['.', 'S'], ['.', '.'],
                                           ['.', 'M'], ['.', '>'], ['v', '.'], ['^', '.'], ['>', '.'], ['M', '.']]))

        constraint.append(TableConstraint(bottom_left[0], bottom_left[1],
                                          [['S', '.'], ['.', 'v'], ['.', '^'], ['.', '<'], ['.', 'S'], ['.', '.'],
                                           ['.', 'M'], ['.', '>'], ['v', '.'], ['^', '.'], ['>', '.'], ['M', '.']]))

        constraint.append(TableConstraint(top[0], top[1],
                                          [['S', '.'], ['.', 'v'], ['.', '>'], ['.', 'S'], ['.', '.'], ['v', '^'],
                                           ['v', 'M'], ['>', '.'], ['M', 'M'], ['M', '^'], ['^', '.']]))

        constraint.append(TableConstraint(bottom[0], bottom[1],
                                          [['S', '.'], ['.', '^'], ['.', '>'], ['.', 'S'], ['.', '.'], ['^', 'v'],
                                           ['^', 'M'], ['>', '.'], ['M', 'M'], ['M', 'v'], ['v', '.']]))

    else:  # cell that does not touch the edge of the board
        constraint.append(TableConstraint(right[0], right[1],
                                          [['S', '.'], ['.', 'v'], ['.', '<'], ['.', 'S'], ['.', '.'], ['<', '>'],
                                           ['<', 'M'], ['v', '.'], ['^', '.'], ['.', '^'], ['M', '.'], ['.', 'M'],
                                           ['>', '.'], ['M', 'M'], ['M', '>']]))

        constraint.append(TableConstraint(left[0], left[1],
                                          [['S', '.'], ['.', 'v'], ['.', '>'], ['.', 'S'], ['.', '.'], ['>', '<'],
                                           ['>', 'M'], ['v', '.'], ['^', '.'], ['.', '^'], ['M', '.'], ['.', 'M'],
                                           ['<', '.'], ['M', 'M'], ['M', '<']]))

        constraint.append(TableConstraint(top[0], top[1],
                                          [['S', '.'], ['.', 'v'], ['.', '>'], ['.', '<'], ['.', 'S'], ['.', '.'],
                                           ['v', '^'], ['v', 'M'], ['>', '.'], ['M', 'M'], ['M', '^'], ['^', '.'],
                                           ['<', '.'], ['M', '.'], ['.', 'M']]))

        constraint.append(TableConstraint(bottom[0], bottom[1],
                                          [['S', '.'], ['.', '^'], ['.', '>'], ['.', '<'], ['.', 'S'], ['.', '.'],
                                           ['^', 'v'], ['^', 'M'], ['>', '.'], ['M', 'M'], ['M', 'v'], ['v', '.'],
                                           ['<', '.'], ['M', '.'], ['.', 'M']]))

        constraint.append(TableConstraint(top_right[0], top_right[1],
                                          [['S', '.'], ['.', 'v'], ['.', '^'], ['.', '<'], ['.', 'S'], ['.', '.'],
                                           ['.', 'M'], ['.', '>'], ['v', '.'], ['^', '.'], ['<', '.'], ['M', '.'],
                                           ['>', '.']]))

        constraint.append(TableConstraint(top_left[0], top_left[1],
                                          [['S', '.'], ['.', 'v'], ['.', '^'], ['.', '<'], ['.', 'S'], ['.', '.'],
                                           ['.', 'M'], ['.', '>'], ['v', '.'], ['^', '.'], ['>', '.'], ['M', '.'],
                                           ['<', '.']]))

        constraint.append(TableConstraint(bottom_left[0], bottom_left[1],
                                          [['S', '.'], ['.', 'v'], ['.', '^'], ['.', '<'], ['.', 'S'], ['.', '.'],
                                           ['.', 'M'], ['.', '>'], ['v', '.'], ['^', '.'], ['>', '.'], ['M', '.'],
                                           ['<', '.']]))

        constraint.append(TableConstraint(bottom_right[0], bottom_right[1],
                                          [['S', '.'], ['.', 'v'], ['.', '^'], ['.', '<'], ['.', 'S'], ['.', '.'],
                                           ['.', 'M'], ['.', '>'], ['v', '.'], ['^', '.'], ['<', '.'], ['M', '.'],
                                           ['>', '.']]))
    return constraint


def check_ship_constraint(csp, ship_cons):
    variables = csp.variables()
    size = int(math.sqrt(len(csp.variables())))
    result_board = [[None for _ in range(size)] for _ in range(size)]  # The solution board after GAC is enforced

    for var in variables:
        var_pos = copy.copy(int(var.name()))
        col = var_pos % size
        row = var_pos // size
        result_board[row][col] = var.getValue()

    # check 'M' state ship
    for y in range(size):
        for x in range(size):
            # check M not at edges
            if result_board[y][x] == 'M' and y not in [0, size - 1] and x not in [0, size - 1]:
                # vertical ship
                if result_board[y][x - 1] == '.' and result_board[y][x + 1] == '.':
                    if result_board[y - 1][x] in ['M', '^'] and result_board[y + 1][x] in ['M', 'v']:
                        if result_board[y - 1][x] == 'M' and result_board[y + 1][x] != 'v':
                            return
                        if result_board[y + 1][x] == 'M' and result_board[y - 1][x] != '^':
                            return
                    else:
                        return
                # horizontal ship
                elif result_board[y - 1][x] == '.' and result_board[y + 1][x] == '.':
                    if result_board[y][x - 1] in ['M', '<'] and result_board[y][x + 1] in ['M', '>']:
                        if result_board[y][x - 1] == 'M' and result_board[y][x + 1] != '>':
                            return
                        if result_board[y][x + 1] == 'M' and result_board[y][x - 1] != '<':
                            return
                    else:
                        return

    # Now check if solution adheres to the ship constraint
    ship_1x1, ship_1x2, ship_1x3, ship_1x4 = ship_cons[0], ship_cons[1], ship_cons[2], ship_cons[3]
    count_1x1, count_1x2, count_1x3, count_1x4 = 0, 0, 0, 0

    for y in range(size):
        for x in range(size):
            if (count_1x1 > ship_1x1) or (count_1x2 > ship_1x2) or (count_1x3 > ship_1x3) or (count_1x4 > ship_1x4):
                return  # return if any ship constraints are violated
            # check for 1x1
            if result_board[y][x] == 'S':
                count_1x1 += 1
            # check for 1x2
            elif result_board[y][x] == '^' and result_board[y + 1][x] == 'v':
                count_1x2 += 1
            elif result_board[y][x] == '<' and result_board[y][x + 1] == '>':
                count_1x2 += 1
            # check for 1x3
            elif result_board[y][x] == 'M':
                if result_board[y][x - 1] == '<' and result_board[y][x + 1] == '>':
                    count_1x3 += 1
                elif result_board[y - 1][x] == '^' and result_board[y + 1][x] == 'v':
                    count_1x3 += 1
            # check for 1x4
            elif result_board[y][x] == 'M' and result_board[y][x + 1] == 'M' and result_board[y][x - 1] == '<':
                count_1x4 += 1
            elif result_board[y][x] == 'M' and result_board[y - 1][x] == '^' and result_board[y + 1][x] == 'M':
                count_1x4 += 1

    if (count_1x1 > ship_1x1) or (count_1x2 > ship_1x2) or (count_1x3 > ship_1x3) or (count_1x4 > ship_1x4):
        return  # final check if any ship constraints are violated
    csp.solutions.append(result_board)


def GACEnforce(constraints, assigned_var, assigned_val, csp):
    while constraints:
        cnstr = constraints.pop(0)
        for var in cnstr.scope():
            # check each item satisfy the constraint
            for val in var.curDomain():
                if not cnstr.hasSupport(var, val):
                    var.pruneValue(val, assigned_var, assigned_val)
                    if var.curDomainSize() == 0:
                        return 'DWO'
                    for recheck in csp.constraintsOf(var):
                        if (recheck != cnstr) and (not recheck in constraints):
                            constraints.append(recheck)
    return 'OK'


def GAC(unassignedVars, ship_const, csp):
    if not unassignedVars:  # Base case: all variables are assigned
        check_ship_constraint(csp, ship_const)  # Among all the solutions, find the one that matches with the ship constr
        return

    var = unassignedVars.pop(0)  # select the next variable to assign
    for val in var.curDomain():
        var.setValue(val)
        no_DWO = True
        if GACEnforce(csp.constraintsOf(var), var, val, csp) == 'DWO':
            no_DWO = False
        if no_DWO:
            GAC(unassignedVars, ship_const, csp)
        Variable.restoreValues(var, val)  # restore values pruned by var = val assignment
    var.setValue(None)  # set var to be unassigned and return to list
    unassignedVars.append(var)
    return


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

    row_con, col_con, ship_con, board = read_input(args.inputfile)
    unassigned_vars, var_dict = create_variable(board)
    constraint = create_constraints(board, row_con, col_con, var_dict)
    csp = CSP('battleship', unassigned_vars, constraint)
    GAC(csp.variables(), ship_con, csp)

    # Output the solution
    sys.stdout = open(args.outputfile, 'w')
    for sol in csp.solutions:
        for row in sol:
            sys.stdout.writelines(row)
            sys.stdout.write("\n")

    sys.stdout.close()
