import numpy as np
from typing import Tuple, Optional, Union, Set, Dict, Any
from fractions import Fraction
import random
from collections import deque
import sys
import os
import json

from misc import *
from genetic_algorithm.individual import Individual
from neural_network import FeedForwardNetwork, linear, sigmoid, tanh, relu



class Vision(object):
    __slots__ = ('dist_to_wall', 'dist_to_apple', 'dist_to_self')
    def __init__(self,
                 dist_to_wall: Union[float, int],
                 dist_to_apple: Union[float, int],
                 dist_to_self: Union[float, int]
                 ):
        self.dist_to_wall = float(dist_to_wall)
        self.dist_to_apple = float(dist_to_apple)
        self.dist_to_self = float(dist_to_self)

class DrawableVision(object):
    __slots__ = ('wall_location', 'apple_location', 'self_location')
    def __init__(self,
                wall_location: Point,
                apple_location: Optional[Point] = None,
                self_location: Optional[Point] = None,
                ):
        self.wall_location = wall_location
        self.apple_location = apple_location
        self.self_location = self_location


class Snake(Individual):
    def __init__(self, board_size: Tuple[int, int],
                 chromosome: Optional[Dict[str, List[np.ndarray]]] = None,
                 start_pos: Optional[Point] = None,  # @TODO: None/Point(10,10),31,10
                 apple_seed: Optional[int] = None,
                 initial_velocity: Optional[str] = None,
                 starting_direction: Optional[str] = None,  # @TODO: 'd'/None
                 hidden_layer_architecture: Optional[List[int]] = [1123125, 9]
                 ):

        self._direction_to_angle = {
            'r': 0.0,
            'u': 90.0,
            'l': 180.0,
            'd': 270.0
        }
    
        self.score = 0  # Number of apples snake gets
        self._fitness = 0  # Overall fitness
        self._frames = 0  # Number of frames that the snake has been alive
        self._frames_since_last_apple = 0
        self.possible_directions = ('u', 'd', 'l', 'r')

        self.board_size = board_size
        self.hidden_layer_architecture = hidden_layer_architecture

        if not start_pos:
            x = random.randint(10, self.board_size[0] - 9)
            y = random.randint(10, self.board_size[1] - 9)
            start_pos = Point(x, y)
        self.start_pos = start_pos

        self._vision_type = VISION_8
        self._vision: List[Vision] = [None] * len(self._vision_type)
        # This is just used so I can draw and is not actually used in the NN
        self._drawable_vision: List[DrawableVision] = [None] * len(self._vision_type)

        # Setting up network architecture
        # Each "Vision" has 3 distances it tracks: wall, apple and self
        # there are also one-hot encoded direction and one-hot encoded tail direction,
        # each of which have 4 possibilities.
        num_inputs = len(self._vision_type) * 3 + 4 + 4
        self.vision_as_array: np.ndarray = np.zeros((num_inputs, 1))
        self.network_architecture = [num_inputs]                          # Inputs
        self.network_architecture.extend(self.hidden_layer_architecture)  # Hidden layers
        self.network_architecture.append(4)                               # 4 outputs, ['u', 'd', 'l', 'r']
        self.network = FeedForwardNetwork(self.network_architecture, relu, sigmoid)

        # If chromosome is set, take it
        if chromosome:
            self._chromosome = chromosome
            self.decode_chromosome()
        else:
            self._chromosome = {}
            self.encode_chromosome()


        # For creating the next apple
        if apple_seed is None:
            apple_seed = np.random.randint(-1000000000, 1000000000)
        self.apple_seed = apple_seed  # Only needed for saving/loading replay
        self.rand_apple = random.Random(self.apple_seed)

        self.apple_location = None
        if starting_direction:
            starting_direction = starting_direction[0].lower()
        else:
            starting_direction = self.possible_directions[random.randint(0, 3)]

        self.starting_direction = starting_direction  # Only needed for saving/loading replay
        self.init_snake(self.starting_direction)
        self.initial_velocity = initial_velocity
        self.init_velocity(self.starting_direction, self.initial_velocity)
        self.generate_apple()

    @property
    def fitness(self):
        return self._fitness
    
    def calculate_fitness(self):
        # Give positive minimum fitness for roulette wheel selection
        # self._fitness = 200*self.score + .25*self._frames
        # if self.score == 0:
        #     self._fitness = .25 * self._frames
        # else:
        #     self._fitness = .25*self._frames * ((2 ** self.score) + (self.score * 1200)) - (self._frames * 100)
        self._fitness = (self._frames) + ((2**self.score) + (self.score**2.1)*500) - (((.25 * self._frames)**1.3) * (self.score**1.2))
        self._fitness = max(self._fitness, 0)

    @property
    def chromosome(self):
        return self._chromosome

    def encode_chromosome(self):
        # L = len(self.network.params) // 2
        L = len(self.network.layer_nodes)
        # Encode weights and bias
        for layer in range(1, L):
            l = str(layer)
            self._chromosome['W' + l] = self.network.params['W' + l].flatten()
            self._chromosome['b' + l] = self.network.params['b' + l].flatten()

    def decode_chromosome(self):
        # L = len(self.network.params) // 2
        L = len(self.network.layer_nodes)
        # Decode weights and bias
        for layer in range(1, L):
            l = str(layer)
            w_shape = (self.network_architecture[layer], self.network_architecture[layer-1])
            b_shape = (self.network_architecture[layer], 1)
            self.network.params['W' + l] = self._chromosome['W' + l].reshape(w_shape)
            self.network.params['b' + l] = self._chromosome['b' + l].reshape(b_shape)

    def look(self):
        # Look all around
        for i, slope in enumerate(self._vision_type):
            vision, drawable_vision = self.look_in_direction(slope)
            self._vision[i] = vision
            self._drawable_vision[i] = drawable_vision
        
        # Update the input array
        self._vision_as_input_array()


    def look_in_direction(self, slope: Slope) -> Tuple[Vision, DrawableVision]:
        dist_to_wall = None
        dist_to_apple = np.inf
        dist_to_self = np.inf

        wall_location = None
        apple_location = None
        self_location = None

        position = self.snake_array[0].copy()
        distance = abs(slope.rise) + abs(slope.run)
        total_distance = 0.0
        # Can't start by looking at yourself
        position.x += slope.run
        position.y += slope.rise
        total_distance += distance
        body_found = False  # Only need to find the first occurance since it's the closest
        food_found = False  # Although there is only one food, stop looking once you find it

        # Keep going until the position is out of bounds
        while self._within_wall(position):
            if not body_found and self._is_body_location(position):
                dist_to_self = total_distance
                self_location = position.copy()
                body_found = True
            if not food_found and self._is_apple_location(position):
                dist_to_apple = total_distance
                apple_location = position.copy()
                food_found = True

            wall_location = position
            position.x += slope.run
            position.y += slope.rise
            total_distance += distance
        assert(total_distance != 0.0)


        # Special case if it can move 3 (i.e. slope is 2)
        if distance == 3:
            # @TODO: The distance may need to be changed since the slope can run past the edge of the board
            pass

        # Normalize where 0 is really far, 1 is really close
        # @TODO: Will it matter that total_distance here can't be 1?
        # @TODO: Could also change it to either:
        #            total_distance += 1   or
        #            normalize with distance as numerator
        dist_to_wall = 1.0 / total_distance
        # dist_to_apple = 1.0 / dist_to_apple
        dist_to_apple = 1.0 if dist_to_apple != np.inf else 0.0
        # dist_to_self = 1.0 / dist_to_self
        dist_to_self = 1.0 if dist_to_self != np.inf else 0.0
        # if dist_to_apple == np.inf:
        #     dist_to_apple = 0.0
        # else:
        #     dist_to_apple = 1.0 / dist_to_apple

        # if dist_to_self == np.inf:
        #     dist_to_self = 0.0
        # else:
        #     dist_to_self = 1.0 / dist_to_self

        vision = Vision(dist_to_wall, dist_to_apple, dist_to_self)
        drawable_vision = DrawableVision(wall_location, apple_location, self_location)
        return (vision, drawable_vision)

    def _vision_as_input_array(self) -> None:
        # Split _vision into np array where rows [0-2] are _vision[0].dist_to_wall, _vision[0].dist_to_apple, _vision[0].dist_to_self,
        # rows [3-5] are _vision[1].dist_to_wall, _vision[1].dist_to_apple, _vision[1].dist_to_self, etc. etc. etc.
        for va_index, v_index in zip(range(0, len(self._vision) * 3, 3), range(len(self._vision))):
            vision = self._vision[v_index]
            self.vision_as_array[va_index, 0]     = vision.dist_to_wall
            self.vision_as_array[va_index + 1, 0] = vision.dist_to_apple
            self.vision_as_array[va_index + 2, 0] = vision.dist_to_self

        i = len(self._vision) * 3  # Start at the end

        direction = self.direction[0].lower()
        # One-hot encode direction
        direction_one_hot = np.zeros((len(self.possible_directions), 1))
        direction_one_hot[self.possible_directions.index(direction), 0] = 1
        self.vision_as_array[i: i + len(self.possible_directions)] = direction_one_hot

        i += len(self.possible_directions)

        # One-hot tail direction
        tail_direction_one_hot = np.zeros((len(self.possible_directions), 1))
        tail_direction_one_hot[self.possible_directions.index(self.tail_direction), 0] = 1
        self.vision_as_array[i: i + len(self.possible_directions)] = tail_direction_one_hot

    def _within_wall(self, position: Point) -> bool:
        return position.x >= 0 and position.y >= 0 and \
               position.x < self.board_size[0] and \
               position.y < self.board_size[1]

    def generate_apple(self) -> None:
        width = self.board_size[0]
        height = self.board_size[1]
        # Find all possible points where the snake is not currently
        possibilities = [divmod(i, height) for i in range(width * height) if divmod(i, height) not in self.snake_array]
        if possibilities:
            loc = self.rand_apple.choice(possibilities)
            self.apple_location = Point(loc[0], loc[1])
        else:
            # I guess you win?
            print('you won!')
            pass

    def init_snake(self, starting_direction: str) -> None:
        """
        Initialize teh snake.
        starting_direction: ('u', 'd', 'l', 'r')
            direction that the snake should start facing. Whatever the direction is, the head
            of the snake will begin pointing that way.
        """        
        head = self.start_pos
        # Body is below
        if starting_direction == 'u':
            snake = [head, Point(head.x, head.y + 1), Point(head.x, head.y + 2)]
        # Body is above
        elif starting_direction == 'd':
            snake = [head, Point(head.x, head.y - 1), Point(head.x, head.y - 2)]
        # Body is to the right
        elif starting_direction == 'l':
            snake = [head, Point(head.x + 1, head.y), Point(head.x + 2, head.y)]
        # Body is to the left
        elif starting_direction == 'r':
            snake = [head, Point(head.x - 1, head.y), Point(head.x - 2, head.y)]

        self.snake_array = deque(snake)
        self._body_locations = set(snake)
        self.is_alive = True

    def update(self):
        if self.is_alive:
            self._frames += 1
            self.look()
            self.network.feed_forward(self.vision_as_array)
            self.direction = self.possible_directions[np.argmax(self.network.out)]
            return True
        else:
            return False

    def move(self) -> bool:
        if not self.is_alive:
            return False

        direction = self.direction[0].lower()
        # Is the direction valid?
        if direction not in self.possible_directions:
            return False
        
        # Find next position
        # tail = self.snake_array.pop()  # Pop tail since we can technically move to the tail
        head = self.snake_array[0]
        if direction == 'u':
            next_pos = Point(head.x, head.y - 1)
        elif direction == 'd':
            next_pos = Point(head.x, head.y + 1)
        elif direction == 'r':
            next_pos = Point(head.x + 1, head.y)
        elif direction == 'l':
            next_pos = Point(head.x - 1, head.y)

        # Is the next position we want to move valid?
        if self._is_valid(next_pos):
            self.snake_array.appendleft(next_pos)
            self._body_locations.update({next_pos})  # Add next position to the set
            # If we just consumed the apple, generate a new one.
            # No need to pop the tail of the snake since the snake is growing here
            if next_pos == self.apple_location:
                self.score += 1
                self._frames_since_last_apple = 0
                self.generate_apple()
            else:
                tail = self.snake_array.pop()
                self._body_locations.symmetric_difference_update({tail})  # Remove tail from the set

            # Figure out which direction the tail is moving
            p2 = self.snake_array[-2]
            p1 = self.snake_array[-1]
            diff = p2 - p1
            if diff.x < 0:
                self.tail_direction = 'l'
            elif diff.x > 0:
                self.tail_direction = 'r'
            elif diff.y > 0:
                self.tail_direction = 'd'
            elif diff.y < 0:
                self.tail_direction = 'u'

            self._frames_since_last_apple += 1
            #@TODO: MAybe make max number of a variable
            if self._frames_since_last_apple > 150:
                self.is_alive = False
                return False

            return True
        else:
            self.is_alive = False
            return False

    def _is_apple_location(self, position: Point) -> bool:
        return position == self.apple_location

    def _is_body_location(self, position: Point) -> bool:
        return position in self._body_locations

    def _is_valid(self, position: Point) -> bool:
        """
        Determine whether a given position is valid.
        Return True if the position is on the board and does not intersect the snake.
        Return False otherwise
        """
        if (position.x < 0) or (position.x > self.board_size[0] - 1):
            return False
        if (position.y < 0) or (position.y > self.board_size[1] - 1):
            return False

        if position == self.snake_array[-1]:
            return True
        # If the position is a body location, not valid.
        # @NOTE: _body_locations will contain tail, so need to check tail first
        elif position in self._body_locations:
            return False
        # Otherwise you good
        else:
            return True

    def init_velocity(self, starting_direction, initial_velocity: Optional[str] = None) -> None:
        if initial_velocity:
            self.direction = initial_velocity[0].lower()
        # Whichever way the starting_direction is
        else:
            self.direction = starting_direction

        # Tail starts moving the same direction
        self.tail_direction = self.direction

def save_snake(population_folder: str, individual_name: str, snake: Snake, settings: Dict[str, Any]) -> None:
    # Make population folder if it doesn't exist
    if not os.path.exists(population_folder):
        os.makedirs(population_folder)

    # Save off settings
    if 'settings.json' not in os.listdir(population_folder):
        f = os.path.join(population_folder, 'settings.json')
        with open(f, 'w', encoding='utf-8') as out:
            json.dump(settings, out, sort_keys=True, indent=4)

    # Make directory for the individual
    individual_dir = os.path.join(population_folder, individual_name)
    os.makedirs(individual_dir)

    # Save some constructor information for replay
    # @NOTE: No need to save chromosome since that is saved as .npy
    # @NOTE: No need to save board_size or hidden_layer_architecture
    #        since these are taken from settings
    constructor = {}
    constructor['start_pos'] = snake.start_pos.to_dict()
    constructor['apple_seed'] = snake.apple_seed
    constructor['initial_velocity'] = snake.initial_velocity
    constructor['starting_direction'] = snake.starting_direction
    snake_constructor_file = os.path.join(individual_dir, 'constructor_params.json')
    print(constructor)
    # Save
    with open(snake_constructor_file, 'w', encoding='utf-8') as out:
        json.dump(constructor, out, sort_keys=True, indent=4)

    L = len(snake.network.layer_nodes)
    for l in range(1, L):
        w_name = 'W' + str(l)
        b_name = 'b' + str(l)

        weights = snake.network.params[w_name]
        bias = snake.network.params[b_name]

        np.save(os.path.join(individual_dir, w_name), weights)
        np.save(os.path.join(individual_dir, b_name), bias)

def load_snake(population_folder: str, individual_name: str, settings: Optional[Dict[str, Any]] = None) -> Snake:
    if not settings:
        f = os.path.join(population_folder, 'settings.json')
        if not os.path.exists(f):
            raise Exception("settings needs to be passed as an argument if 'settings.json' does not exist under population folder")
        
        with open(f, 'r', encoding='utf-8') as fp:
            settings = json.load(fp)

    params = {}
    for fname in os.listdir(os.path.join(population_folder, individual_name)):
        extension = fname.rsplit('.npy', 1)
        if len(extension) == 2:
            param = extension[0]
            params[param] = np.load(os.path.join(population_folder, individual_name, fname))
        else:
            continue

    # Load constructor params for the specific snake
    constructor_params = {}
    snake_constructor_file = os.path.join(population_folder, individual_name, 'constructor_params.json')
    with open(snake_constructor_file, 'r', encoding='utf-8') as fp:
        constructor_params = json.load(fp)

    snake = Snake(settings['board_size'], chromosome=params, 
                  start_pos=Point.from_dict(constructor_params['start_pos']),
                  apple_seed=constructor_params['apple_seed'],
                  initial_velocity=constructor_params['initial_velocity'],
                  starting_direction=constructor_params['starting_direction'],
                  hidden_layer_architecture=settings['hidden_network_architecture'])
    return snake