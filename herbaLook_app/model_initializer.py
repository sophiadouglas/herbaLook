import tensorflow as tf
from .utils import get_current_datetime
from six.moves import cPickle




# Time for print logs
CURRENT_TIME_STAMP = get_current_datetime()

model_session = None

def initialize_model(graph_filepath):
    # Load the frozen model 
    with tf.gfile.GFile(graph_filepath,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        print(CURRENT_TIME_STAMP, "Frozen model loaded")
     
    # Load graphs definition (nodes)
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def)
        print(CURRENT_TIME_STAMP, "Graphs defined")


    # Prepare input tensors (name are subjected to different networks)
    field_tensor_input = graph.get_tensor_by_name('import/Placeholder:0')
    field_tensor_feat_norm = graph.get_tensor_by_name('import/field_embedding/l2_normalize:0') # HFTL 2023
    # field_tensor_feat_norm = graph.get_tensor_by_name('import/l2_normalize:0') # HFTL 2022
    field_tensor_last_layer = graph.get_tensor_by_name('import/field/field/Conv2d_7b_1x1/Relu:0')
    print(CURRENT_TIME_STAMP, "Tensors defined")

    # Create a TensorFlow session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(graph=graph, config=config)    

    # Create a TensorFlow session and return it
    return sess, field_tensor_input, field_tensor_feat_norm, field_tensor_last_layer





def load_herbarium_dictionary(herbarium_dictionary_pkl_path):
    # Load herbarium dictionary
    with open(herbarium_dictionary_pkl_path,'rb') as fpkl:
        herbarium_dictionary = cPickle.load(fpkl)
        print(CURRENT_TIME_STAMP, "Herbarium dictionary loaded")
    return herbarium_dictionary


