# -*- coding: utf-8 -*-
"""
Created on 2018-10-30 14:05:21

@author: Xu Kaiqiang
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape)  # (60000, 28, 28)
# print(x_test.shape)   # (10000, 28, 28)
print(y_train.shape)
print(np.unique(y_test))
x_train, x_test = x_train / 255.0, x_test / 255.0

eval_len = 1000
x_train, x_vali = x_train[:-eval_len], x_train[-eval_len:]
y_train, y_vali = y_train[:-eval_len], y_train[-eval_len:]


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=5, validation_data=(x_vali, y_vali))
"""
loss: 0.0438 - acc: 0.9856 - val_loss: 0.0653 - val_acc: 0.9820
"""
# print(model.evaluate(x_test, y_test))
"""
test_loss: 0.06450289927392733, test_acc: 0.9807
"""

import neat
import numpy as np

x_train, x_vali, x_test = x_train.reshape((-1, 28*28)), x_vali.reshape((-1, 28*28)), x_test.reshape((-1, 28*28))
def accuracy(y, y_predict):
    y = np.array(y)
    y_predict = np.array(y_predict)
    assert y.shape[0] == y_predict.shape[0]
    hit_count = 0
    for one_label, one_y_predict in zip(y, y_predict):
        import pdb
        # pdb.set_trace()
        if one_label == np.argmax(one_y_predict):
            import pdb
            # pdb.set_trace()
            hit_count += 1
    return hit_count * 1.0 / y.shape[0]

# evaluation function
def eval_genomes(genomes, config):
    # import pdb
    # pdb.set_trace()
    print("come here to eval genomes.....")
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        y_out = []
        for x in x_vali:
            y_out.append(net.activate(x))
        # loss = tf.Session().run(tf.nn.softmax_cross_entropy_with_logits_v2(
        #     logits=y_out, labels=tf.one_hot(y_vali, depth=10)
        # ))
        # genome.fitness = - sum(loss) / len(loss)
        genome.fitness = accuracy(y_vali, y_out)
        print('this genome fitness: ', genome_id, genome.fitness)


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




y_out = list(map(lambda x: winner_net.activate(x), x_vali))
vali_loss = tf.Session().run(tf.nn.softmax_cross_entropy_with_logits(
            logits=y_out, labels=tf.one_hot(y_vali, depth=10)
        ))
vali_acc = accuracy(y_vali, y_out)
print("validation loss: ", sum(vali_loss) / len(vali_loss), "validation acc: ", vali_acc)


# y_out = list(map(lambda x: winner_net.activate(x), x_test))
# test_loss = tf.Session().run(tf.nn.softmax_cross_entropy_with_logits(
#             logits=y_out, labels=tf.one_hot(y_test, depth=10)
#         ))
# test_acc = accuracy(y_test, y_out)
# print("test loss: ", sum(test_loss) / len(test_loss), "test acc: ", test_acc)
