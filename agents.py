import numpy as np
import random as rand
import matplotlib.pyplot as plt


def make_board(size, mines):
    board = np.full([size, size], 0)
    # setting mines randomly
    for i in range(mines):
        random_x = rand.randint(0, size-1)
        random_y = rand.randint(0, size-1)
        while board[random_x][random_y] == -1:
            random_x = rand.randint(0, size-1)
            random_y = rand.randint(0, size-1)
        board[random_x][random_y] = -1
    # setting the numbers for each cell depending on how many mines surround it
    # checking all eight sides surrounding a cell
    for i in range(size):
        for j in range(size):
            if board[i][j] != -1:
                board[i][j] = 0
                if i > 0 and board[i-1][j] == -1:
                    board[i][j] += 1
                if i < size-1 and board[i+1][j] == -1:
                    board[i][j] += 1
                if j > 0 and board[i][j-1] == -1:
                    board[i][j] += 1
                if j < size-1 and board[i][j+1] == -1:
                    board[i][j] += 1
                if i > 0 and j > 0 and board[i-1][j-1] == -1:
                    board[i][j] += 1
                if i < size-1 and j < size-1 and board[i+1][j+1] == -1:
                    board[i][j] += 1
                if i > 0 and j < size-1 and board[i-1][j+1] == -1:
                    board[i][j] += 1
                if i < size-1 and j > 0 and board[i+1][j-1] == -1:
                    board[i][j] += 1
    return board
#print the board
def print_board(board, size):
    print('-'*(size*4), end="")
    print()
    for x in board:
        for y in x:
            if y == -1:
                cell = 'M'
            else:
                cell = y
            print(f'| {cell} ', end="")
        print("|")
        print('-'*(size*4))

# gets neighbors of a cell
def get_neighbors(cell, n):
    i, j = cell
    neighbors = []
    if i > 0:
        neighbors.append((i-1, j))
    if i < n-1:
        neighbors.append((i+1, j))
    if j > 0:
        neighbors.append((i, j-1))
    if j < n-1:
        neighbors.append((i, j+1))
    if i > 0 and j > 0:
        neighbors.append((i-1, j-1))
    if i < n-1 and j < n-1:
        neighbors.append((i+1, j+1))
    if i > 0 and j < n-1:
        neighbors.append((i-1, j+1))
    if i < n-1 and j > 0:
        neighbors.append((i+1, j-1))
    return neighbors

##################################################################################
# BASIC AGENT ####################################################################
##################################################################################   

def basic_agent(board, size):
    knowledge_base = []
    # this holds the info of each cell
    cell_info = {}
    # list of cells available to query found through inference
    next_cell = []
    # mines found
    found_mines = 0
    # score = successfully found mines through inference
    score = 0
    while len(cell_info) != size**2:
        # cell to be queried next
        query_cell = (0, 0)
        # used to remove unnecessary equations
        new_knowledge_base = []
        # decision step
        # if a non-mine cell has been infered, it will be used as the next cell to query
        # otherwise, a random cell is picked
        if next_cell:
            query_cell = next_cell.pop()
        else:
            random_x = rand.randint(0, size-1)
            random_y = rand.randint(0, size-1)
            while (random_x, random_y) in cell_info:
                random_x = rand.randint(0, size-1)
                random_y = rand.randint(0, size-1)
            query_cell = (random_x, random_y)
        cell_value = board[query_cell[0]][query_cell[1]]
        if cell_value == -1:
            found_mines += 1
            cell_info[query_cell] = -1
        else:
            cell_info[query_cell] = 0
            equation = get_neighbors(query_cell, size)
            equation = (equation, cell_value)
            knowledge_base.append(equation)
        # inference step
        new_knowledge_base = []
        for index, equation in enumerate(knowledge_base):
            flag = False
            var, val = equation
            new_var = []
            for i in var:
                if i in cell_info:
                    val += cell_info[i]
                    continue
                new_var.append(i)
            if new_var:
                flag = True
                var = new_var
                equation = (var, val)
                knowledge_base[index] = (var, val)
            # if number of cells and the value of the equation is same, all cells are mines 
            if len(var) == val:
                for i in var:
                    flag = False
                    if i not in cell_info:
                        cell_info[i] = -1
                        score += 1
                        found_mines += 1
            # if the value of the equation is zero, all cells are non-mines
            if val == 0:
                for i in var:
                    flag = False
                    if i not in cell_info:
                        cell_info[i] = 0
                        next_cell.append(i)
            # to keep the equation or not
            if flag:
                new_knowledge_base.append(equation)
        knowledge_base = new_knowledge_base
    return score


##################################################################################################
# Performance of basic agent #####################################################################
##################################################################################################
'''
size = 20
dataX = []
dataY = []
runs = 30

for mines in range(1, size**2):
    print(mines)
    board = make_board(size, mines)
    sum_of_scores = 0
    for run in range(runs):
        print(run, end=' ')
        sum_of_scores += (basic_agent(board, size)/mines)
    print(sum_of_scores/runs)
    avg_scores = (sum_of_scores) / runs
    mine_density = mines/size**2
    dataX.append(mine_density)
    dataY.append(avg_scores)
print('here')
plt.plot(dataX, dataY)
#plt.title('Basic agent performace')
#plt.ylabel('Average score')
#plt.xlabel('Mine density')
#plt.show()
'''
##################################################################################
# BASIC AGENT with improved selection ############################################
##################################################################################   

def basic_agentv2(board, size):
    knowledge_base = []
    # this holds the info of each cell
    cell_info = {}
    # list of cells available to query found through inference
    next_cell = []
    # mines found
    found_mines = 0
    # score = successfully found mines through inference
    score = 0
    while len(cell_info) != size**2:
        # cell to be queried next
        query_cell = (0, 0)
        # used to remove unnecessary equations
        new_knowledge_base = []
        # decision step
        # if a non-mine cell has been infered, it will be used as the next cell to query
        # otherwise, a random cell is picked
        if next_cell:
            query_cell = next_cell.pop()
        elif knowledge_base:
            length = len(knowledge_base)
            random_eq = rand.randint(0, length-1)
            eq_length = len(knowledge_base[random_eq][0])
            random_var = rand.randint(0, eq_length-1)
            while knowledge_base[random_eq][0][random_var] in cell_info:
                random_eq = rand.randint(0, length-1)
                eq_length = len(knowledge_base[random_eq][0])
                random_var = rand.randint(0, eq_length-1)
            query_cell = knowledge_base[random_eq][0][random_var]
        else:
            random_x = rand.randint(0, size-1)
            random_y = rand.randint(0, size-1)
            while (random_x, random_y) in cell_info:
                random_x = rand.randint(0, size-1)
                random_y = rand.randint(0, size-1)
            query_cell = (random_x, random_y)
        cell_value = board[query_cell[0]][query_cell[1]]
        if cell_value == -1:
            found_mines += 1
            cell_info[query_cell] = -1
        else:
            cell_info[query_cell] = 0
            equation = get_neighbors(query_cell, size)
            equation = (equation, cell_value)
            knowledge_base.append(equation)
        # inference step
        new_knowledge_base = []
        for index, equation in enumerate(knowledge_base):
            flag = False
            var, val = equation
            new_var = []
            for i in var:
                if i in cell_info:
                    val += cell_info[i]
                    continue
                new_var.append(i)
            if new_var:
                flag = True
                var = new_var
                equation = (var, val)
                knowledge_base[index] = (var, val)
            # if number of cells and the value of the equation is same, all cells are mines 
            if len(var) == val:
                for i in var:
                    flag = False
                    if i not in cell_info:
                        cell_info[i] = -1
                        score += 1
                        found_mines += 1
            # if the value of the equation is zero, all cells are non-mines
            if val == 0:
                for i in var:
                    flag = False
                    if i not in cell_info:
                        cell_info[i] = 0
                        next_cell.append(i)
            # to keep the equation or not
            if flag:
                new_knowledge_base.append(equation)
        knowledge_base = new_knowledge_base
    return score


##################################################################################################
# Performance of basic agentv2 ###################################################################
##################################################################################################
'''
size = 20
dataX = []
dataY = []
runs = 30

for mines in range(1, size**2):
    print(mines)
    board = make_board(size, mines)
    sum_of_scores = 0
    for run in range(runs):
        print(run, end=' ')
        sum_of_scores += (basic_agentv2(board, size)/mines)
    print(sum_of_scores/runs)
    avg_scores = (sum_of_scores) / runs
    mine_density = mines/size**2
    dataX.append(mine_density)
    dataY.append(avg_scores)
print('here')
plt.plot(dataX, dataY)
#plt.title('Basic agent vs Basic agent with better selection')
#plt.ylabel('Average score')
#plt.xlabel('Mine density')
#plt.show()
'''
##################################################################################
# IMPROVED AGENT #################################################################
##################################################################################

def subset_solver(equation_one, equation_two):
    a, b = equation_one
    c, d = equation_two
    set_one = set(a)
    set_two = set(c)
    if set_one > set_two:
        set_three = set_one - set_two
        val = b - d
        return (list(set_three), val)
    if set_two > set_one:
        set_three = set_two - set_one
        val = d - b
        return (list(set_three), val)
    return False

def improved_agent(board, size):
    knowledge_base = []
    # this holds the info of each cell
    cell_info = {}
    # list of cells available to query found through inference
    next_cell = []
    # mines found
    found_mines = 0
    # score = successfully found mines through inference
    score = 0
    while len(cell_info) != size**2:
        #print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
        #print("knowlege base: ", end='')
        #print(knowledge_base)
        #print("Known safe cells: ", end='')
        #print(next_cell)
        #print("Cell info: ", end="")
        #print(cell_info)
        # cell to be queried next
        query_cell = (0, 0)
        # used to remove unnecessary equations
        new_knowledge_base = []
        # decision step
        # if a non-mine cell has been infered, it will be used as the next cell to query
        # otherwise, a random cell is picked
        if next_cell:
            query_cell = next_cell.pop()
        else:
            random_x = rand.randint(0, size-1)
            random_y = rand.randint(0, size-1)
            while (random_x, random_y) in cell_info:
                random_x = rand.randint(0, size-1)
                random_y = rand.randint(0, size-1)
            query_cell = (random_x, random_y)
        #print('chosen cell: '+str(query_cell))
        cell_value = board[query_cell[0]][query_cell[1]]
        if cell_value == -1:
            found_mines += 1
            cell_info[query_cell] = -1
        else:
            cell_info[query_cell] = 0
            equation = get_neighbors(query_cell, size)
            equation = (equation, cell_value)
            knowledge_base.append(equation)
        additional_eqs = []
        # subset solver
        for equation_one in knowledge_base:
            for equation_two in knowledge_base:
                new_eq = subset_solver(equation_one, equation_two)
                if new_eq and new_eq not in knowledge_base and new_eq not in additional_eqs:
                     additional_eqs.append(new_eq)
        knowledge_base.extend(additional_eqs)
        # inference step
        new_knowledge_base = []
        for index, equation in enumerate(knowledge_base):
            flag = False
            var, val = equation
            new_var = []
            for i in var:
                if i in cell_info:
                    val += cell_info[i]
                    continue
                new_var.append(i)
            if new_var:
                flag = True
                var = new_var
                equation = (var, val)
                knowledge_base[index] = (var, val)
            # if number of cells and the value of the equation is same, all cells are mines 
            if len(var) == val:
                for i in var:
                    flag = False
                    if i not in cell_info:
                        cell_info[i] = -1
                        score += 1
                        found_mines += 1
            # if the value of the equation is zero, all cells are non-mines
            if val == 0:
                for i in var:
                    flag = False
                    if i not in cell_info:
                        cell_info[i] = 0
                        next_cell.append(i)
            # to keep the equation or not
            if flag:
                new_knowledge_base.append(equation)
        knowledge_base = new_knowledge_base
    return score

##################################################################################################
# Performance of improved agent ##################################################################
##################################################################################################
'''
size = 20
dataX = []
dataY = []
runs = 30

for mines in range(1, size**2):
    print(mines)
    board = make_board(size, mines)
    sum_of_scores = 0
    for run in range(runs):
        print(run, end=' ')
        sum_of_scores += (improved_agent(board, size)/mines)
    print(sum_of_scores/runs)
    avg_scores = (sum_of_scores) / runs
    mine_density = mines/size**2
    dataX.append(mine_density)
    dataY.append(avg_scores)
print('here')
plt.plot(dataX, dataY)
#plt.title('Improved agent performace (ICE)')
#plt.ylabel('Average score')
#plt.xlabel('Mine density')
#plt.legend(["Basic agent", "Improved agent"])
#plt.show()
'''
##################################################################################
# IMPROVED AGENT with improved selection #########################################
##################################################################################

# does not randomly choose a cell instead uses randomly chooses an unknown variable from the knowledge base
def improved_agentv2(board, size):
    knowledge_base = []
    # this holds the info of each cell
    cell_info = {}
    # list of cells available to query found through inference
    next_cell = []
    # mines found
    found_mines = 0
    # score = successfully found mines through inference
    score = 0
    while len(cell_info) != size**2:
        print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------')
        print("knowledge base: ", end='')
        print(knowledge_base)
        print("Known safe cells: ", end='')
        print(next_cell)
        print("Cell info: ", end="")
        print(cell_info)
        # cell to be queried next
        query_cell = (0, 0)
        # used to remove unnecessary equations
        new_knowledge_base = []
        # decision step
        # if a non-mine cell has been infered, it will be used as the next cell to query
        # otherwise, a the first unknown cell from knowledge is picked
        # and if knowledge base is empty, it will choose a cell randomly
        if next_cell:
            query_cell = next_cell.pop()
        elif knowledge_base:
            length = len(knowledge_base)
            random_eq = rand.randint(0, length-1)
            eq_length = len(knowledge_base[random_eq][0])
            random_var = rand.randint(0, eq_length-1)
            while knowledge_base[random_eq][0][random_var] in cell_info:
                random_eq = rand.randint(0, length-1)
                eq_length = len(knowledge_base[random_eq][0])
                random_var = rand.randint(0, eq_length-1)
            query_cell = knowledge_base[random_eq][0][random_var]
        else:
            random_x = rand.randint(0, size-1)
            random_y = rand.randint(0, size-1)
            while (random_x, random_y) in cell_info:
                random_x = rand.randint(0, size-1)
                random_y = rand.randint(0, size-1)
            query_cell = (random_x, random_y)
        print("Queried cell:",query_cell)
        cell_value = board[query_cell[0]][query_cell[1]]
        if cell_value == -1:
            found_mines += 1
            cell_info[query_cell] = -1
        else:
            cell_info[query_cell] = 0
            equation = get_neighbors(query_cell, size)
            equation = (equation, cell_value)
            knowledge_base.append(equation)
        # subset solver
        additional_eqs = []
        for equation_one in knowledge_base:
            for equation_two in knowledge_base:
                new_eq = subset_solver(equation_one, equation_two)
                if new_eq and new_eq not in knowledge_base and new_eq not in additional_eqs:
                     additional_eqs.append(new_eq)
        knowledge_base.extend(additional_eqs)
        # inference step
        new_knowledge_base = []
        for index, equation in enumerate(knowledge_base):
            flag = False
            var, val = equation
            new_var = []
            for i in var:
                if i in cell_info:
                    val += cell_info[i]
                    continue
                new_var.append(i)
            if new_var:
                flag = True
                var = new_var
                equation = (var, val)
                knowledge_base[index] = (var, val)
            # if number of cells and the value of the equation is same, all cells are mines 
            if len(var) == val:
                for i in var:
                    flag = False
                    if i not in cell_info:
                        cell_info[i] = -1
                        score += 1
                        found_mines += 1
            # if the value of the equation is zero, all cells are non-mines
            if val == 0:
                for i in var:
                    flag = False
                    if i not in cell_info:
                        cell_info[i] = 0
                        next_cell.append(i)
            # to keep the equation or not
            if flag:
                new_knowledge_base.append(equation)
        knowledge_base = new_knowledge_base
    return score

##################################################################################################
# Performance of improved agent v2################################################################
##################################################################################################
'''
size = 20
dataX = []
dataY = []
runs = 30

for mines in range(1, size**2):
    print(mines)
    board = make_board(size, mines)
    sum_of_scores = 0
    for run in range(runs):
        print(run, end=' ')
        sum_of_scores += (improved_agentv2(board, size)/mines)
    print(sum_of_scores/runs)
    avg_scores = (sum_of_scores) / runs
    mine_density = mines/size**2
    dataX.append(mine_density)
    dataY.append(avg_scores)
print('here')
plt.plot(dataX, dataY)
plt.title('Better selection mechanism comparision')
plt.ylabel('Average score')
plt.xlabel('Mine density')
plt.legend(["Basic", "Basic v2", "Improved", "Improved v2"])
plt.show()
'''
size = 4
mines = 7
board = make_board(size, mines)
print(improved_agentv2(board, size))