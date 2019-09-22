from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt
import sys
from typing import List
from neural_network import *
from snake import Snake


class NeuralNetworkViz(QtWidgets.QWidget):
    def __init__(self, parent, snake: Snake):
        super().__init__(parent)
        self.snake = snake
        self.horizontal_distance_between_layers = 50
        self.vertical_distance_between_nodes = 10
        self.num_neurons_in_largest_layer = max(self.snake.network.layer_nodes)
        # self.setFixedSize(600,800)
        self.neuron_locations = {}
        self.show()


    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter()
        painter.begin(self)

        self.show_network(painter)
        
        painter.end()

    def update(self) -> None:
        self.repaint()

    def show_network(self, painter: QtGui.QPainter):
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        vertical_space = 8
        radius = 8
        height = self.frameGeometry().height()
        width = self.frameGeometry().width()
        layer_nodes = self.snake.network.layer_nodes

        default_offset = 30
        h_offset = default_offset
        inputs = self.snake.vision_as_array
        out = self.snake.network.feed_forward(inputs)  # @TODO: shouldnt need this
        max_out = np.argmax(out)
        
        # Draw nodes
        for layer, num_nodes in enumerate(layer_nodes):
            v_offset = (height - ((2*radius + vertical_space) * num_nodes))/2
            activations = None
            if layer > 0:
                activations = self.snake.network.params['A' + str(layer)]

            for node in range(num_nodes):
                x_loc = h_offset
                y_loc = node * (radius*2 + vertical_space) + v_offset
                t = (layer, node)
                if t not in self.neuron_locations:
                    self.neuron_locations[t] = (x_loc, y_loc + radius)
                
                painter.setBrush(QtGui.QBrush(Qt.white, Qt.NoBrush))
                # Input layer
                if layer == 0:
                    # Is there a value being fed in
                    if inputs[node, 0] > 0:
                        painter.setBrush(QtGui.QBrush(Qt.green))
                    else:
                        painter.setBrush(QtGui.QBrush(Qt.white))
                # Hidden layers
                elif layer > 0 and layer < len(layer_nodes) - 1:
                    try:
                        saturation = max(min(activations[node, 0], 1.0), 0.0)
                    except:
                        print(self.snake.network.params)
                        import sys
                        sys.exit(-1)
                    painter.setBrush(QtGui.QBrush(QtGui.QColor.fromHslF(125/239, saturation, 120/240)))
                # Output layer
                elif layer == len(layer_nodes) - 1:
                    text = ('U', 'D', 'L', 'R')[node]
                    painter.drawText(h_offset + 30, node * (radius*2 + vertical_space) + v_offset + 1.5*radius, text)
                    if node == max_out:
                        painter.setBrush(QtGui.QBrush(Qt.green))
                    else:
                        painter.setBrush(QtGui.QBrush(Qt.white))

                painter.drawEllipse(x_loc, y_loc, radius*2, radius*2)
            h_offset += 150

        # Reset horizontal offset for the weights
        h_offset = default_offset

        # Draw weights
        # For each layer starting at 1
        for l in range(1, len(layer_nodes)):
            weights = self.snake.network.params['W' + str(l)]
            prev_nodes = weights.shape[1]
            curr_nodes = weights.shape[0]
            # For each node from the previous layer
            for prev_node in range(prev_nodes):
                # For all current nodes, check to see what the weights are
                for curr_node in range(curr_nodes):
                    # If there is a positive weight, make the line blue
                    if weights[curr_node, prev_node] > 0:
                        painter.setPen(QtGui.QPen(Qt.blue))
                    # If there is a negative (impeding) weight, make the line red
                    else:
                        painter.setPen(QtGui.QPen(Qt.red))
                    # Grab locations of the nodes
                    start = self.neuron_locations[(l-1, prev_node)]
                    end = self.neuron_locations[(l, curr_node)]
                    # Offset start[0] by diameter of circle so that the line starts on the right of the circle
                    painter.drawLine(start[0] + radius*2, start[1], end[0], end[1])