import cv2
import skimage
import numpy as np
import pyautogui
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sudokusolver import SudokuSolver
import time

def main():
    pyautogui.hotkey("alt", "tab", interval=0.1)
    screenshot = pyautogui.screenshot()
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    thresh = preprocess(screenshot)
    sudoku_contour = find_sudoku(thresh)
    cropped_sudoku = get_sudoku(screenshot, sudoku_contour)
    squares_images = split_grid(cropped_sudoku)
    sudoku_vec = squares_images_to_sudoku(squares_images)
    sudoku = SudokuSolver(sudoku_vec)
    sudoku_solved = sudoku.solve()
    solve_on_website(sudoku_contour, sudoku_solved)


def preprocess(screenshot):
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)[1] #cv2.THRESH_OTSU
    return thresh


def find_sudoku(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    square = find_squares(contours)
    square = sorted(square, key=cv2.contourArea, reverse=True)
    return square[0]
    

def find_squares(contours):
    square = []
    for contorno in contours:
        epsilon = cv2.arcLength(contorno, True)
        approx = cv2.approxPolyDP(contorno, 0.01 * epsilon, True)
        _, _, w, h = cv2.boundingRect(approx)
        aspect_ratio = w/float(h)
        if len(approx) == 4 and abs(aspect_ratio - 1) < 0.1:
            contour = approx
            square.append(contour)
    return square


def get_sudoku(screenshot, sudoku_contour):
    x, y, w, h = cv2.boundingRect(sudoku_contour)
    cropped = screenshot[y:y+h, x:x+w]
    return cropped


def split_grid(cropped_sudoku):
    img = preprocess(cropped_sudoku)
    img = skimage.segmentation.clear_border(img)
    img = 255 - img
    h,_ = img.shape
    square_size = h // 9
    squares = []
    for i in range(9):
        for j in range(9):
            square_img = img[i * square_size : (i + 1) * square_size, j * square_size : (j + 1) * square_size]
            square_img = cv2.resize(square_img, dsize = (45, 45), interpolation=cv2.INTER_AREA)
            squares.append(square_img)
    return squares


def squares_images_to_sudoku(squares_images):
    knn = create_knn_model()
    sudoku = np.zeros((81), dtype=int)
    for i, image in enumerate(squares_images):
        sudoku[i] = predict_digit(image, knn)
    return sudoku.reshape(9,9)


def predict_digit(img, knn):
    img_vec = img.reshape(1, -1)
    prediction = knn.predict(img_vec)[0]
    return prediction


def create_knn_model():
    df = pd.read_csv("dataset.csv")
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x, y)
    return knn


def solve_on_website(sudoku_contour, solved):
    x, y, w, h = cv2.boundingRect(sudoku_contour)
    square_size = h // 9
    print(solved)
    for i in range(9):
        for j in range(9):
            pyautogui.click(x + j*square_size + square_size//2, y + i*square_size + square_size//2, _pause=False)
            pyautogui.press(str(solved[i,j]), _pause=False)
        

main()
