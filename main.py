from actor_critic_nn import AC_Network
from worker import Worker
from create_game import Game
import tensorflow as tf
import multiprocessing
import threading
import os
from vizdoom import *
from time import sleep
import argparse

parser = argparse.ArgumentParser(description='A3C-Doom')
parser.add_argument("name", type=str,
                    help="run id")
parser.add_argument("max_local_episodes", type=int,
                    help="max number of episodes ran by EACH thread")
parser.add_argument("-thread_num", type=int,
                    help="number threads")

args = parser.parse_args()
max_local_episodes = args.max_local_episodes # max number of episodes of EACH thread

max_episode_length = 300
gamma = .99 # discount rate for advantage estimation and reward discounting
s_size = 7056 # Observations are greyscale frames of 84 * 84 * 1
a_size = 3 # Agent can move Left, Right, or Fire
load_model = False
model_path = './' + args.name + '/model'
frame_path = './' + args.name + '/frames'

tf.reset_default_graph()
if not os.path.exists(model_path):
    os.makedirs(model_path)
#Create a directory to save episode playback gifs to
if not os.path.exists(frame_path):
    os.makedirs(frame_path)

with tf.device("/cpu:0"):
    # global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = AC_Network(s_size,a_size,'global',None) # Generate global network
    # num_workers = multiprocessing.cpu_count() # Set workers to number of available CPU threads
    num_workers = args.thread_num
    workers = []
    # Create worker classes
    for i in range(num_workers):
        game = Game()
        workers.append(
            Worker(
                game.health_gathering(),
                i,
                s_size,
                a_size,
                trainer,
                model_path,
                frame_path
                # global_episodes
            )
        )
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        
    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(
            max_episode_length,
            gamma,
            sess,
            coord,
            saver,
            max_local_episodes
        )
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)	
