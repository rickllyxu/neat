# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
mnist = tf.keras.datasets.mnist
import random
# from __future__ import print_function
import neat

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images_sum = 60000
train_images = train_images.reshape((train_images_sum, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images_sum = 10000
test_images = test_images.reshape((test_images_sum, 28 * 28))
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

def hit(label, input, genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    output = net.activate(input)
    result = output.index(max(output))
    if (result == label):
        return True
    else:
        return False

def eval_genomes(genomes, config):
    generation = random.randint(0, 1000)
    # print(generation, genomes)
    batch_size = 20
    for genome_id, genome in genomes:
        hitCount = 0

        for i in range(generation * batch_size % train_images_sum,
                      (generation + 1) * batch_size % train_images_sum):
            mnist_inputs = train_images[i]

            if hit(train_labels[i], mnist_inputs, genome, config):
                hitCount += 1
        genome.fitness = hitCount / batch_size

# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-mnist')

# reset result file
# res = open("result.txt", "w")
# res.close()

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
    if (hit(test_labels[i], test_images[i], winner_net)):
        hitCount += 1
print("hit {0} of {1}".format(hitCount, len(test_labels)))
# visualize.draw_net(config, winner, True, node_names=node_names)

"""
# Show output of the most fit genome against training data.
print('\nOutput:')
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
for xi, xo in zip(xor_inputs, xor_outputs):
    output = winner_net.activate(xi)
    print("  input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
"""