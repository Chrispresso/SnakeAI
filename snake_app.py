from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt
import sys
from typing import List
from snake import *
import numpy as np
SQUARE_SIZE = (16, 16)


class SnakeWidget(QtWidgets.QWidget):
    def __init__(self, board_size=(50, 50)):
        super().__init__()
        self.board_size = board_size
        self.setFixedSize(SQUARE_SIZE[0] * self.board_size[0], SQUARE_SIZE[1] * self.board_size[1])
        self.new_game()

        self.draw_vision = True

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(1000./15)
        self.show()

    def new_game(self) -> None:
        self.snake = Snake(self.board_size, seed=0)
    
    def update(self):
        if self.snake.is_alive:
            self.snake.move()
        else:
            print('dead')
            import sys
            sys.exit(-1)
        self.repaint()

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
            angle = (self.snake._direction_to_angle[self.snake.direction]) % 360
            start = self.snake.snake_array[0]
            start_x = start.x * SQUARE_SIZE[0] + SQUARE_SIZE[0]/2
            start_y = start.y * SQUARE_SIZE[1] + SQUARE_SIZE[1]/2  
            end_x = 50 * np.cos(angle*np.pi/180.) + start.x
            end_y = 50 * -np.sin(angle*np.pi/180.) + start.y
            end_x = end_x * SQUARE_SIZE[0] + SQUARE_SIZE[0]/2
            end_y = end_y * SQUARE_SIZE[1] + SQUARE_SIZE[1]/2
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
    ex = SnakeWidget()
    sys.exit(app.exec_())