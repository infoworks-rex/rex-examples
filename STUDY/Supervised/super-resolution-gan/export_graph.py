""" Freeze variables and convert 2 generator networks to 2 GraphDef files.
This makes file size smaller and can be used for inference in production.
An example of command-line usage is:
python export_graph.py --checkpoint_dir checkpoints/20170424-1152 \
                       --XtoY_model apple2orange.pb \
                       --YtoX_model orange2apple.pb \
                       --image_size 256
"""

import tensorflow as tf
import os
from tensorflow.python.tools.freeze_graph import freeze_graph
from model import * 
#import utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('checkpoint_dir', 'modelGAN', 'checkpoints directory path')
tf.flags.DEFINE_string('model', 'SR_freeze.pb', 'model name, ')



def export_graph(model_name):
  graph = tf.Graph()

  with graph.as_default():
    x = tf.placeholder(tf.float32, [1, 256, 256, 3])
    y =buildESRGAN_g(x,isTraining=False)

    g_vars = [x for x in tf.trainable_variables() if "ESRGAN_g" in x.name]    
    
    output_image = tf.identity(y, name='output_image')
    restore_saver = tf.train.Saver()
    export_saver = tf.train.Saver()

  with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    latest_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    restore_saver.restore(sess, latest_ckpt)
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [output_image.op.name])

    tf.train.write_graph(output_graph_def, 'pretrained', model_name, as_text=False)

def main(unused_argv):
  export_graph(FLAGS.model)

if __name__ == '__main__':
  tf.app.run()
