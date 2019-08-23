settings = {
    'hidden_layer_activation':     'relu',     # Options are [relu, sigmoid, tanh, linear, leaky_relu]
    'output_layer_activation':     'sigmoid',  # Options are [relu, sigmoid, tanh, linear, leaky_relu]
    'hidden_network_architecture': [9, 9],     # A list containing number of nodes in each hidden layer
    'vision_type':                 8,          # Number of directions the snake can see in

    #### GA stuff ####
    ## Mutation ##
    # Mutation rate is the probability that a given gene in a chromosome will randomly mutate
    'mutation_rate':               0.05,       # Value must be between [0.00, 1.00)
    # If the mutation rate type is static, then the mutation rate will always be `mutation_rate`,
    # otherwise if it is decaying it will decrease as the number of generations increase
    'mutation_rate_type':          'static',   # Options are [static, decaying]
    # The type of mutation to perform
    'mutation_type':               'gaussian',  # Options are [gaussian]
    ## Population ##
    # Number of individuals in the population
    'population_size':             20,
    # Number of top performing individuals to copy over from previous generation
    # Should be a small number
    'num_elitism':                 1,
    ## Crossover ##
    # The type of crossover to perform
    'crossover_type':              'SBX',       # Options are [SBX]
    ## Selection ##
    # Selection type determines the way in which we select individuals for crossover
    'selection_type':              'roulette_wheel'
}