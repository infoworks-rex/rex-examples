import tensorflow as tf
from tensorflow.python.tools import freeze_graph

freeze_graph.freeze_graph(input_graph='./graph.pbtxt',
                          input_saver="",
                          input_binary=False,
                          input_checkpoint='./DnCNN-tensorflow-353500',
                          output_node_names='sub',
                          restore_op_name="",
                          filename_tensor_name="",
                          output_graph='./dncnn353500.pb',
                          clear_devices=False,
                          initializer_nodes="")

