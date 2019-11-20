from utility_functions import *
import numpy as np

'''from google.colab import drive
drive.mount('/content/drive')
filepath = '/content/drive/My Drive/WORK/rnn-workshop/bird-genera.txt'
'''
#use this when running locally:
filepath = './villages.txt'

# Read input data and find basic info like vocab_size, data_size etc
data, chars, data_size, vocab_size = read_data(filepath)
print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

#Utilities that make sampling easier
char_to_index, index_to_char = sampling_utils(chars)

print("Index to character mapping:\n",index_to_char)
print("Character to Index mapping (Reverse of Index to Character mapping):\n",char_to_index)


#Complete this function 
def forward_pass_one_element(parameters, a_prev, x):
    
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    a_next = np.tanh(np.dot(Wax, x) + np.dot(Waa,a_prev) + b) # complete this line
    p_t = np.dot(Wya, a_next) + by      # unnormalized log probabilities for next chars
    p_t = softmax(p_t)                  # probabilities for next chars

    return a_next, p_t

def rnn_forward_one_sequence(X, Y, a0, parameters, vocab_size = 27):
    
    # Initialize x, a and y_hat as empty dictionaries
    x, a, y_hat = {}, {}, {}
    a[-1] = np.copy(a0)
    # initialize your loss to 0
    loss = 0
    
    for t in range(len(X)):
        # Set x[t] to be the one-hot vector representation of the t'th character in X.
        # if X[t] == None, we just have x[t]=0.
        # This is used to set the input for the first timestep to the zero vector.
        x[t] = np.zeros((vocab_size,1)) 
        if (X[t] != None):
            x[t][X[t]] = 1
        
        # Run one step forward of the RNN
        a[t], y_hat[t] = forward_pass_one_element(parameters, a[t - 1], x[t])
        
        # Update the loss by substracting the cross-entropy term of this time-step from it.
        loss -= np.log(y_hat[t][Y[t],0])
        
    cache = (y_hat, a, x)
        
    return loss, cache

def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):

    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    da = np.dot(parameters['Wya'].T, dy) + gradients['da_next'] # backprop into h
    daraw = (1 - a * a) * da # backprop through tanh nonlinearity
    gradients['db'] += daraw
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
    return gradients


#Complete this function
def initialize_parameters(n_a, n_x, n_y):
    np.random.seed(1)
    Wax = np.random.randn(n_a, n_x)*0.01 # input to hidden
    Waa = np.random.randn(n_a, n_a)*0.01 # hidden to hidden
    Wya = np.random.randn(n_y, n_a)*0.01 # hidden to output
    b = np.zeros((n_a, 1)) # hidden bias
    by = np.zeros((n_y, 1)) # output bias

    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b,"by": by}

    return parameters

def update_parameters(parameters, gradients, lr):

    parameters['Wax'] += - lr*gradients['dWax'] # complete this line
    parameters['Waa'] += - lr*gradients['dWaa'] # complete this line
    parameters['Wya'] += - lr*gradients['dWya'] # complete this line
    parameters['b']  += - lr*gradients['db'] # complete this line
    parameters['by']  += - lr*gradients['dby'] # complete this line

    return parameters

# Complete this function
def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
    # Forward propagate through time
    loss, cache = rnn_forward_one_sequence(X, Y, a_prev, parameters, vocab_size)

    # Backpropagate through time
    gradients, a = rnn_backward(X, Y, parameters, cache)

    # Clip your gradients between -5 (min) and 5 (max)
    gradients = clip_to_plus_minus_5(gradients) #{key:np.clip(gradient, -5, 5, gradient) for key,gradient in gradients.items() }

    # Update parameters
    parameters = update_parameters(parameters, gradients, learning_rate)
    return loss, gradients, a[len(X)-1]

def model(data, index_to_char, char_to_index, num_iterations = 2000, n_a = 50, bird_names = 15, vocab_size = 27, lr = 0.01):
    # Retrieve n_x and n_y from vocab_size
    n_x, n_y = vocab_size, vocab_size

    # Initialize parameters
    parameters = initialize_parameters(n_a, n_x, n_y)

    # Initialize loss
    loss = get_initial_loss(vocab_size, bird_names)

    # Build list of all bird names (training examples).
    with open(filepath) as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    # Shuffle list of all bird names
    np.random.seed(0)
    np.random.shuffle(examples)

    # Initialize the hidden state of your RNN
    a_prev = np.zeros((n_a, 1))

    # Optimization loop
    for j in range(num_iterations):
        # Use the hint above to define one training example (X,Y)
        index = j % len(examples)
        X = [None] + [char_to_index[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_index["\n"]]

        # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
        # Choose a learning rate of 0.01
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate = lr)

        # Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
        loss = smooth(loss, curr_loss)

        # Every check_pt iterations,
        # generate "n" characters thanks to sample() to check if the model is learning properly
        check_pt = 3000
        if j %  check_pt == 0:
            generate_sequences(bird_names, char_to_index, index_to_char, j, loss, parameters)

    return parameters


def generate_sequences(bird_names, char_to_index, index_to_char, j, loss, parameters):
    print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
    # The number of bird names to print
    seed = 1
    for name in range(bird_names):
        # Sample indices and print them
        sampled_indices = sample(parameters, char_to_index, seed)
        print_sample(sampled_indices, index_to_char)
        seed += 1
    print('\n')


def rnn_backward(X, Y, parameters, cache):
    # Initialize gradients as an empty dictionary
    gradients = {}

    # Retrieve from cache and parameters
    (y_hat, a, x) = cache
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']

    # each one should be initialized to zeros of the same dimension as its corresponding parameter
    gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
    gradients['db'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])

    # Backpropagate through time
    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1
        gradients = rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t-1])
    return gradients, a

parameters = model(data, index_to_char, char_to_index, num_iterations=10, vocab_size=vocab_size)


print_sample(sample(parameters, char_to_index, 200), index_to_char)
