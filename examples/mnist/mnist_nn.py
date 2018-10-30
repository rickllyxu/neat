import keras
keras.__version__

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import numpy as np

# from __future__ import print_function
import neat

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

eval_len = 10
eval_image = train_images[:eval_len]
eval_labels = train_labels[:eval_len]

"""
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_acc:', test_acc)

"""


# 2-input XOR inputs and expected outputs.
# mnist_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
# mnist_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        fitness = [10.0] * eval_len
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for i in range(0, eval_len):
            minist_inputs = eval_image[i]
            mnist_outputs = [0.0] * 10
            mnist_outputs[eval_labels[i]] = 1.0
            output = net.activate(minist_inputs)
            for j in range(0, 10):
                fitness[i] -= (output[j] - mnist_outputs[j]) ** 2
            output.sort()
            fitness[i] += output[9] - output[8]
        genome.fitness = np.mean(fitness)

# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-mnist')

# reset result file
res = open("result.txt", "w")
res.close()

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(False))

# Run until a solution is found.
winner = p.run(eval_genomes)

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))

# print results on evaluate set
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
for i in range(0, 10):
    output = winner_net.activate(eval_image[i])
    fitness = 10;
    mnist_outputs = [0.0] * 10
    mnist_outputs[eval_labels[i]] = 1.0
    for j in range(0, 10):
        fitness -= (output[j] - mnist_outputs[j]) ** 2
    print(eval_labels[i], fitness)
    print("got {!r}".format(output))

# test on test dataset
hitCount = 0
for i in range(0, len(test_labels)):
    output = winner_net.activate(test_images[i])
    result = output.index(max(output))
    if (result == test_labels[i]):
        hitCount += 1
print("hit {0} of {1}".format(hitCount, len(eval_labels)))
# visualize.draw_net(config, winner, True, node_names=node_names)

"""
# Show output of the most fit genome against training data.
print('\nOutput:')
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
for xi, xo in zip(xor_inputs, xor_outputs):
    output = winner_net.activate(xi)
    print("  input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
"""