from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt
import sys
from typing import List
from snake import *
import numpy as np
from nn_viz import NeuralNetworkViz
from neural_network import FeedForwardNetwork, sigmoid, linear, relu


SQUARE_SIZE = (16, 16)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, board_size=(50, 50)):
        super().__init__()
        self.board_size = board_size
        self.border = (10, 10)
        self.snake_widget_width = SQUARE_SIZE[0] * self.board_size[0]
        self.snake_widget_height = max(SQUARE_SIZE[1] * self.board_size[1], 800)

        self.top = 150
        self.left = 150
        self.width = self.snake_widget_width + 600 + 2*self.border[0]
        self.height = self.snake_widget_height + 2*self.border[1]
        self.ff = FeedForwardNetwork([8*3+8,12,9,4], sigmoid, linear)
        self.snake = Snake(board_size, seed=0)

        self.init_window()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        # self.timer.start(1000./15)

        self.show()
        self.update()

    def init_window(self):
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setWindowTitle('Snake AI')
        self.setGeometry(self.top, self.left, self.width, self.height)

        # Create the Neural Network window
        self.nn_viz_window = NeuralNetworkViz(self.centralWidget, self.ff, self.snake)
        self.nn_viz_window.setGeometry(QtCore.QRect(0, 0, 600, self.snake_widget_height + 2*self.border[1]))
        self.nn_viz_window.setObjectName('nn_viz_window')

        # Create SnakeWidget window
        self.snake_widget_window = SnakeWidget(self.centralWidget, self.board_size, self.snake)
        self.snake_widget_window.setGeometry(QtCore.QRect(600 + self.border[0], self.border[1], self.snake_widget_width, self.snake_widget_height))
        self.snake_widget_window.setObjectName('snake_widget_window')

    def update(self) -> None:
        self.snake_widget_window.update()
        self.nn_viz_window.update()


class SnakeWidget(QtWidgets.QWidget):
    def __init__(self, parent, board_size=(50, 50), snake=None):
        super().__init__(parent)
        self.board_size = board_size
        # self.setFixedSize(SQUARE_SIZE[0] * self.board_size[0], SQUARE_SIZE[1] * self.board_size[1])
        self.new_game()
        if snake:
            self.snake = snake
        self.setFocus()

        self.draw_vision = True
        self.show()

    def new_game(self) -> None:
        self.snake = Snake(self.board_size, seed=0)
    
    def update(self):
        if self.snake.is_alive:
            self.snake.update()
            self.repaint()
        else:
            print('dead')
            import sys
            sys.exit(-1)

    def draw_border(self, painter: QtGui.QPainter) -> None:
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.setPen(QtGui.QPen(Qt.black))
        width = self.frameGeometry().width()
        height = self.frameGeometry().height()
        painter.drawLine(0, 0, width, 0)
        painter.drawLine(width, 0, width, height)
        painter.drawLine(0, height, width, height)
        painter.drawLine(0, 0, 0, height)

    def draw_snake(self, painter: QtGui.QPainter) -> None:
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        painter.setPen(QtGui.QPen(Qt.black))
        painter.setBrush(QtGui.QBrush(Qt.red))

        for point in self.snake.snake_array:
            painter.drawRect(point.x * SQUARE_SIZE[0],  # Upper left x-coord
                             point.y * SQUARE_SIZE[1],  # Upper left y-coord
                             SQUARE_SIZE[0],            # Width
                             SQUARE_SIZE[1])            # Height

        if self.draw_vision:
            start = self.snake.snake_array[0]

            if self.snake._drawable_vision[0]:
                for drawable_vision in self.snake._drawable_vision:
                    start_x = start.x * SQUARE_SIZE[0] + SQUARE_SIZE[0]/2
                    start_y = start.y * SQUARE_SIZE[1] + SQUARE_SIZE[1]/2
                    if drawable_vision.apple_location:
                        painter.setPen(QtGui.QPen(Qt.green))
                        end_x = drawable_vision.apple_location.x * SQUARE_SIZE[0] + SQUARE_SIZE[0]/2
                        end_y = drawable_vision.apple_location.y * SQUARE_SIZE[1] + SQUARE_SIZE[1]/2
                        painter.drawLine(start_x, start_y, end_x, end_y)
                        start_x, start_y = end_x, end_y
                    if drawable_vision.self_location:
                        painter.setPen(QtGui.QPen(Qt.red))
                        end_x = drawable_vision.self_location.x * SQUARE_SIZE[0] + SQUARE_SIZE[0]/2
                        end_y = drawable_vision.self_location.y * SQUARE_SIZE[1] + SQUARE_SIZE[1]/2 
                        painter.drawLine(start_x, start_y, end_x, end_y)
                        start_x, start_y = end_x, end_y
                    if drawable_vision.wall_location:
                        painter.setPen(QtGui.QPen(Qt.black))
                        end_x = drawable_vision.wall_location.x * SQUARE_SIZE[0] + SQUARE_SIZE[0]/2
                        end_y = drawable_vision.wall_location.y * SQUARE_SIZE[1] + SQUARE_SIZE[1]/2 
                        painter.drawLine(start_x, start_y, end_x, end_y)


    def draw_apple(self, painter: QtGui.QPainter) -> None:
        apple_location = self.snake.apple_location
        if apple_location:
            painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
            painter.setPen(QtGui.QPen(Qt.black))
            painter.setBrush(QtGui.QBrush(Qt.green))

            painter.drawRect(apple_location.x * SQUARE_SIZE[0],
                             apple_location.y * SQUARE_SIZE[1],
                             SQUARE_SIZE[0],
                             SQUARE_SIZE[1])

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter()
        painter.begin(self)

        self.draw_border(painter)
        self.draw_apple(painter)
        self.draw_snake(painter)
        
        painter.end()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key_press = event.key()
        if key_press == Qt.Key_Up:
            self.snake.direction = 'u'
        elif key_press == Qt.Key_Down:
            self.snake.direction = 'd'
        elif key_press == Qt.Key_Right:
            self.snake.direction = 'r'
        elif key_press == Qt.Key_Left:
            self.snake.direction = 'l'



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())