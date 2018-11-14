import tensorflow as tf
import data_queue
import model
import denseNet
import math
import os
from config import log, NUM_VAL

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('output','../data/model',
                          """Directory for event logs and checkpoints""")
tf.app.flags.DEFINE_string('tune_from','',
                          """Path to pre-trained model checkpoint""")
tf.app.flags.DEFINE_string('tune_scope','',
                          """Variable scope for training""")
tf.app.flags.DEFINE_string("val_device", '/cpu:0',
                           """device for validation process.""")
tf.app.flags.DEFINE_integer('batch_size',1,
                            """Mini-batch size""")#32
tf.app.flags.DEFINE_string("val_input_device", "/cpu:0",
                           """Device for validation data preprocess/batching graph placement""")
tf.app.flags.DEFINE_string("val_path", "../data/val/",
                           """Base directory for validating data""")
tf.app.flags.DEFINE_string('filename_pattern','words-*',
                           """File pattern for input data""")
tf.app.flags.DEFINE_integer('num_input_threads',1,    #4
                          """Number of readers for input data""")
tf.app.flags.DEFINE_integer('width_threshold',None,
                            """Limit of input image width""")
tf.app.flags.DEFINE_integer('length_threshold',None,
                            """Limit of input string length width""")
tf.app.flags.DEFINE_integer("num_epochs", None,
                            """number of epochs for input queue""")
tf.app.flags.DEFINE_integer("eval_interval_secs", 300,
                            """Interval seconds to perform evaluation.""")
tf.logging.set_verbosity(tf.logging.INFO)

stepCount= math.ceil(NUM_VAL / FLAGS.batch_size)

def _get_threaded_input(
        data_dir,
        filename_pattern,
        batch_size,
        num_threads,
        input_device,
        num_epochs,
):
    """Set up and return image, label, width and text tensors"""

    image, width, label, length, text, filename = data_queue.threaded_input_pipeline(
        data_dir,
        str.split(filename_pattern, ','),
        batch_size= batch_size,
        num_threads= num_threads,
        num_epochs= num_epochs,  # Repeat for streaming
        batch_device= input_device,
        preprocess_device= input_device)

    return image, width, label, length

def _add_loss(logits, label, sequence_length):
    with tf.name_scope("loss"):
        loss = model.ctc_loss_layer(logits, label, sequence_length)
        tf.summary.scalar("val_loss", loss)
    return loss

def _add_metrics(rnn_logits, sequence_length, label, label_length):
    """Create ops for testing (all scalars):
       loss: CTC loss function value,
       label_error:  Batch-normalized edit distance on beam search max
       sequence_error: Batch-normalized sequence error rate
    """
    with tf.name_scope("metrics"):
        predictions,_ = tf.nn.ctc_beam_search_decoder(rnn_logits,
                                                   sequence_length,
                                                   beam_width=128,
                                                   top_paths=1,
                                                   merge_repeated=True)
        hypothesis = tf.cast(predictions[0], tf.int32) # for edit_distance
        label_errors = tf.edit_distance(hypothesis, label, normalize=False)
        sequence_errors = tf.count_nonzero(label_errors,axis=0)
        total_label_error = tf.reduce_sum( label_errors )
        total_labels = tf.reduce_sum( label_length )
        label_error = tf.truediv( total_label_error,
                                  tf.cast(total_labels, tf.float32 ),
                                  name='label_error')
        sequence_error = tf.truediv( tf.cast( sequence_errors, tf.int32 ),
                                     tf.shape(label_length)[0], # batchsize
                                     name='sequence_error')
        tf.summary.scalar( 'val_label_error', label_error )
        tf.summary.scalar( 'val_sequence_error', sequence_error )

    return label_error, sequence_error

def _get_session_config():
    """Setup session config to soften device placement"""
    from config import CPU_NUM

    config=tf.ConfigProto(
        device_count={"CPU": CPU_NUM},
        inter_op_parallelism_threads = CPU_NUM,
        intra_op_parallelism_threads = CPU_NUM,
        allow_soft_placement=True,
        log_device_placement=False
    )

    return config

def main(argv=None):

    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        # isTraining= tf.placeholder(tf.bool, shape= (), name= "isTraining")
        isTraining= False
        image, width, label, length= _get_threaded_input(
                data_dir=FLAGS.val_path,
                filename_pattern=FLAGS.filename_pattern,
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_input_threads,
            input_device=FLAGS.val_input_device,
            num_epochs= FLAGS.num_epochs,
        )


        with tf.device(FLAGS.val_device):

            # features,sequence_length = model.convnet_layers( image, width, isTraining) # mode: training mode for dropout layer, True for training while False for testing
            features,sequence_length = denseNet.Dense_net( image, width, isTraining)
            logits = model.rnn_layers(features, sequence_length,
                                      data_queue.num_classes())
            with tf.variable_scope(tf.get_variable_scope(),reuse=False): # purpose here?
                loss= _add_loss(logits, label, sequence_length)
                metrics= labelErrors, sequenceErrors = _add_metrics(logits, sequence_length, label, length)

        summary= tf.summary.merge_all()
        init_op = tf.group( tf.global_variables_initializer(),
                            tf.local_variables_initializer())

        # saver= tf.train.Saver(tf.global_variables())
        summaryWriter= tf.summary.FileWriter(logdir= os.path.join(FLAGS.output, 'test'))
        # coordinator= tf.train.Coordinator()
        # session_config = _get_session_config()
        # scaffold= tf.train.Scaffold(
        #     init_op= init_op,
        #     summary_op= summary,
        #     saver= saver,
        # )
        stopHook= tf.contrib.training.StopAfterNEvalsHook(stepCount)
        summaryHook= tf.contrib.training.SummaryAtEndHook(
            summary_writer= summaryWriter,
            summary_op= summary,
        )
        tf.contrib.training.evaluate_repeatedly(
            checkpoint_dir= FLAGS.output,
            eval_ops= [loss, metrics],
            hooks= [stopHook, summaryHook],
            config= _get_session_config(),
            eval_interval_secs=FLAGS.eval_interval_secs

        )

if __name__ == "__main__":
    tf.app.run()