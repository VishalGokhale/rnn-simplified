## These are utility functions. Don't change anything here
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001


def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0 / vocab_size) * seq_length


def sample(parameters, char_to_ix, seed):
    # Retrieve parameters and relevant shapes from "parameters" dictionary
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    # Step 1: Create the one-hot vector x for the first character (initializing the sequence generation).
    x = np.zeros((vocab_size, 1))
    # Step 1': Initialize a_prev as zeros
    a_prev = np.zeros((n_a, 1))

    # Create an empty list of indices, this is the list which will contain the list of indices of the characters to generate
    indices = []

    # Idx is a flag to detect a newline character, we initialize it to -1
    idx = -1

    # Loop over time-steps t. At each time-step, sample a character from a probability distribution and append
    # its index to "indices". We'll stop if we reach 50 characters (which should be very unlikely with a well
    # trained model), which helps debugging and prevents entering an infinite loop.
    counter = 0
    newline_character = char_to_ix['\n']

    while (idx != newline_character and counter != 50):
        # Step 2: Forward propagate x using the equations for an rnn unit
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by

        y = softmax(z)

        # Set Seed
        np.random.seed(counter + seed)

        # Step 3: Sample the index of a character within the vocabulary from the probability distribution y
        # p = np.array([0.1, 0.0, 0.7, 0.2])

        idx = np.random.choice(list(range(vocab_size)), p=y.ravel())

        # Append the index to "indices"
        indices.append(idx)

        # Step 4: Overwrite the input character as the one corresponding to the sampled index.
        x = np.zeros((vocab_size, 1))
        x[idx] = 1

        # Update "a_prev" to be "a"
        a_prev = a

        # update seed and counter
        seed += 1
        counter += 1

    if (counter == 50):
        indices.append(char_to_ix['\n'])

    return indices


def print_sample(sample_index, index_to_char):
    txt = ''.join(index_to_char[index] for index in sample_index)
    txt = txt[0].upper() + txt[1:]  # capitalize first character
    print('%s' % (txt,), end='')

#A quick check to see if the sampling is working as expected

def read_data(filepath):
    #global data, chars, data_size, vocab_size
    data = open(filepath, 'r').read()
    data = data.lower()
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    return data, chars, data_size, vocab_size


def sampling_utils(chars):
    char_to_index = {ch: i for i, ch in enumerate(sorted(chars))}
    index_to_char = {i: ch for i, ch in enumerate(sorted(chars))}
    return char_to_index, index_to_char

def quick_check(filepath):
    data, chars, data_size, vocab_size = read_data(filepath)
    char_to_index, index_to_char = sampling_utils(chars)
    np.random.seed(2)
    n_a = 100
    Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
    b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
    indices = sample(parameters, char_to_index, 0)
    print("Sampling:")
    print("list of sampled indices:\n", indices, "\n")
    # print("list of sampled characters:\n", [index_to_char[i] for i in indices])
    print_sample(indices, index_to_char)

    """### Expected Output
    Sampling:
    list of sampled indices:
    [12, 17, 24, 14, 13, 9, 10, 22, 24, 6, 13, 11, 12, 6, 21, 15, 21, 14, 3, 2, 1, 21, 18, 24, 7, 25, 6, 25, 18, 10, 16, 2, 3, 8, 15, 12, 11, 7, 1, 12, 10, 2, 7, 7, 11, 3, 6, 12, 7, 12, 0] 

    Lqxnmijvxfmklfuouncbaurxgyfyrjpbcholkgaljbggkcflgl
    """

#quick_check("./villages.text")

'''
def clip(gradients, maxValue):

    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
    # clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby]
    for gradient in [dWax, dWaa, dWya, db, dby]:
        gradient = np.clip(a=gradient, amin = -maxValue, a_max=maxValue, out=gradient) # complete this line  . use np.clip()


    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    return gradients

'''

def clip(gradients, maxValue):
    return {key: np.clip(gradient, -maxValue, maxValue, gradient) for key, gradient in gradients.items()}


clip_to_plus_minus_5 = lambda gradients: clip(gradients,5)