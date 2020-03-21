import tensorflow as tf
import importlib
from config import FLAGS
import numpy as np
alex_model = importlib.import_module('models.' + 'alexnet')
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.client import timeline

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()


def run_demo():
    # images = tf.Variable(tf.random_normal([1, 224, 224, 3], dtype=tf.float32, stddev=1e-1))
    images = np.random.rand(1, 224, 224, 3)
    model = alex_model.AlexNet()
    # init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    init = tf.compat.v1.global_variables_initializer()
    device_count = {'GPU': 1} if FLAGS.use_gpu else {'GPU': 0}
    # with tf.Session(config=tf.ConfigProto(device_count=device_count, allow_soft_placement=True)) as sess:
    # with InteractiveSession(config=tf.compat.v1.ConfigProto(device_count=device_count, allow_soft_placement=True)) as sess:
    config = tf.ConfigProto(device_count=device_count)
    config.gpu_options.allow_growth = True
    sess = InteractiveSession(config=config)
    sess.run(init)  
    output_value = sess.run([model.output, ], 
                            feed_dict={model.input_images: images}, 
                            options=run_options, 
                            run_metadata=run_metadata)
    print(output_value)
    saver.save(sess, save_path=FLAGS.save_model_path)
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('./models/timeline.json', 'w') as f:
        f.write(ctf)


if __name__ == '__main__':
    run_demo()
