from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt
import sys
from typing import List
from snake import *

SQUARE_SIZE = (8, 8)


class SnakeWidget(QtWidgets.QWidget):
    def __init__(self, board_size=(50, 50)):
        super().__init__()
        self.board_size = board_size
        self.setFixedSize(SQUARE_SIZE[0] * self.board_size[0], SQUARE_SIZE[1] * self.board_size[1])
        self.new_game()

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