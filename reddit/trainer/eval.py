import model
import tensorflow as tf
import utils


def evaluate(target,
             parameter_servers,
             is_chief,
             checkpoint_directory=None,
             filenames=None,
             sentence_length=128,
             vocab_size=2**15,
             output_dir=None,
             batch_size=1024,
             embedding_size=128,
             num_epochs=2):

    graph = tf.Graph()
    with graph.as_default():
        sentences, scores = model.get_inputs(
            filenames, batch_size, num_epochs, sentence_length)

        lstm = model.BasicRegressionLSTM(
            sentences, scores, parameter_servers, vocab_size, embedding_size)

    tf.contrib.learn.evaluate(
        graph,
        output_dir,
        checkpoint_directory,
        {'loss': lstm.loss},
        global_step_tensor=lstm.global_step,
        supervisor_master=target
    )


if __name__ == "__main__":
    parser = utils.base_parser()
    parser.add_argument(
        '--checkpoint-directory',
        type=utils.gcs_file,
        required=True
    )

    utils.dispatch(
        evaluate,
        **parser.parse_args().__dict__
    )
