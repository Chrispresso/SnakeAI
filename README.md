# Snake AI
# Current Version 1.0
## Licensed under MIT license - so you can use this however you please :)
## Dependencies
There exist three dependencies:

1. numpy
2. pyqt5
3. Python3.6+

To install dependencies, run `pip3 install -r requirements.txt`

## Getting started

1. Clone the repo or download it in someway `git clone https://github.com/Chrispresso/SnakeAI.git`
2. The two places you will ever really need to change stuff unless you feel like going crazy are `settings.py`, which control the hyperparameters of the Neural Network and Genetic Algorithm and `snake_app.py`, which is the graphics and GA.
3. Pick some things you would like to test with under `settings.py`. If you change the `board_size` drastically you will probably want to change `SQUARE_SIZE` under `snake_app.py`. The current settings when you first clone are the settings I used for training a snake to solve 10x10 and the 50x50 grids. Play around with this stuff if you want, you can always create another population in a new command window that use different settings. 
4. Head over to `snake_app.py` and adjust `show=True, fps=200` if you would like. `show` controls whether or not to display the snakes learning. FPS in the case of show is capped at your monitor refresh. Definitely faster to train with `show=False, fps=1000`.
5. Go to the area of `# Next generation` and you can print out the fitness if you would like. This is also where you can `save` the snake. Well you can save anywhere I guess, but this is where I saved the best snake from each generation. This can be done with `save_snake('path/to/population/folder', 'snake_name (i.e. best_snake_gen0)', snake, self.settings)`. This saves the snake, the constructor params that were used to create the snake and the `settings.py` file used for hyperparameters. If you load the same snake you saved, the snake will play **exactly** how it did before. The apple locations are based off an initially `apple seed`. So if you load the same snake without modifying the contructor, then the snake will replay what it did. Very helpful for me since I trained without visualizations and needed to go back and record stuff for the video.
6. Run it! However you like, you can run it and get some snakes generating!

## Loading snakes
Let's say you have a 50 generations of snakes saved and you want to create a new population with the last 10 generations. You could start a new instance of `snake_app.py` and modify `for _ in range(self.settings['num_parents']):` portion to generate 10 less snakes. Then you can load your 10 best snakes and insert them into the population. This is where you can choose to either modify the constructor of your snake to have a different `apple_seed` or allow the snake to run it's previous course. The choice is up to you and totally dependent on your goals!

If the ability to load snakes is something you want to have an easier time with let me know and I can work on that.
