from typing import Tuple, Optional, Union, Set
from fractions import Fraction
import random
from collections import deque
import sys
from misc import *



class Vision(object):
    __slots__ = ('dist_to_wall', 'dist_to_apple', 'dist_to_self')
    def __init__(self,
                 dist_to_wall: Optional[Union[float, int]] = None,
                 dist_to_apple: Optional[Union[float, int]] = None,
                 dist_to_self: Optional[Union[float, int]] = None
                 ):
        self.dist_to_wall = float(dist_to_wall) if dist_to_wall else None
        self.dist_to_apple = float(dist_to_apple) if dist_to_apple else None
        self.dist_to_self = float(dist_to_self) if dist_to_self else None



class Snake(object):
    def __init__(self, board_size: Tuple[int, int],
                 start_pos: Optional[Point] = None,
                 seed: Optional[int] = None,
                 initial_velocity: Optional[str] = None,
                 starting_direction: Optional[str] = None
                 ):

        self._direction_to_angle = {
            'r': 0.0,
            'u': 90.0,
            'l': 180.0,
            'd': 270.0
        }
        self.score = 0
        self.board_size = board_size

        if not start_pos:
            x = random.randint(10, self.board_size[0] - 9)
            y = random.randint(10, self.board_size[1] - 9)
            start_pos = Point(x, y)
        self.start_pos = start_pos

        # For creating the next apple
        self.rand_apple = random.Random(seed)

        self.apple_location = None
        if starting_direction:
            starting_direction = starting_direction[0].lower()
        else:
            starting_direction = ('u', 'd', 'l', 'r')[random.randint(0, 3)]
        self.init_snake(starting_direction)
        self.init_velocity(starting_direction, initial_velocity)
        self.generate_apple()

    def look(self, slope: float):
        pass

    def look_in_direction(self, slope: Slope) -> Vision:
        dist_to_wall = None
        dist_to_apple = None
        dist_to_self = None
        position = self.snake_array[0].copy()
        distance = abs(slope.rise) + abs(slope.run)
        total_distance = 0.0
        # Can't start by looking at yourself
        position.x += slope.run
        position.y += slope.rise
        total_distance += distance
        body_found = False  # Only need to find the first occurance since it's the closest
        # Keep going until the position is out of bounds
        while self._within_wall(position):
            pass

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

    def move(self) -> bool:
        if not self.is_alive:
            return False

        direction = self.direction[0].lower()
        # Is the direction valid?
        if direction not in ('u', 'd', 'l', 'r'):
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

            return True
        else:
            self.is_alive = False
            return False

    def _is_apple_position(self, position: Point) -> bool:
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