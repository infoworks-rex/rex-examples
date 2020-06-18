import tensorflow as tf
from tensorflow.python.tools import freeze_graph

freeze_graph.freeze_graph(input_graph='/home/yoona/Autopilot-TensorFlow/graph.pb',
                          input_saver="",
                          input_binary=True,
                          input_checkpoint='/home/yoona/Autopilot-TensorFlow/save/model.ckpt',
                          output_node_names='Mul',
                          restore_op_name="",
                          filename_tensor_name="",
                          output_graph='./frozen.pb',
                          clear_devices=False,
                          initializer_nodes="")
