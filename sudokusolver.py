import numpy as np
import time

class SudokuSolver:


    def __init__(self, sudoku):
        self.bitmap = np.ones((9,9,9), dtype=bool)
        self.sudoku = np.zeros((9, 9), dtype=int)
        for i in range(9):
            for j in range(9):
                if sudoku[i][j] != 0:
                    self.place_number(i,j, sudoku[i][j])


    def place_number(self, i, j, number):
        self.sudoku[i][j] = number
        self.bitmap[i][j] = np.zeros(9, dtype=bool)
        for k in range(9):
            self.bitmap[i][k][number-1] = False
            self.bitmap[k][j][number-1] = False
        for k in range(3):
            for l in range(3):
                self.bitmap[i//3*3+k][j//3*3+l][number-1] = False


    def solve(self):
        self.backtrack()
        return self.sudoku
    

    def backtrack(self):
        if self.is_solved():
            return self.sudoku
        i, j = self.least_option_cell()
        for number in range (1,10):
            if self.can_place_number(i, j, number):
                curr_bitmap = self.bitmap.copy()
                curr_sudoku = self.sudoku.copy()
                self.place_number(i, j, number)
                if self.backtrack() is not None:
                    return self.sudoku
                self.bitmap = curr_bitmap
                self.sudoku = curr_sudoku
        return None


    def is_solved(self):
        return np.sum(self.sudoku == 0) == 0

    def least_option_cell(self):
        cell_sums = np.sum(self.bitmap, axis=2)
        cell_sums[cell_sums == 0] = 10
        min_sum_indices = np.unravel_index(np.argmin(cell_sums), cell_sums.shape)
        return min_sum_indices
    
    def can_place_number(self, i, j, number):
        return self.bitmap[i][j][number-1]
    
#estaria bien optimizar con "movimientos triviales"