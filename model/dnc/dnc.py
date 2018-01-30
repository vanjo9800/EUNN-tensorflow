import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMStateTuple
import os
from .memory import Memory
from .utility import *

class DNC:
    def __init__(self, controller_class, model, input_size, output_size, max_sequence_length, memory_words_num = 256, memory_word_size = 64, memory_read_heads = 4, batch_size = 1):
        self.input_size = input_size
        self.output_size = output_size
        self.max_sequence_length = max_sequence_length
        self.words_num = memory_words_num
        self.word_size = memory_word_size
        self.read_heads = memory_read_heads
        self.batch_size = batch_size
        self.model = model

        self.memory = Memory(self.words_num, self.word_size, self.read_heads, self.batch_size)
        self.controller = controller_class(self.input_size, self.output_size, self.read_heads, self.word_size, self.batch_size, self.model)
        self.input_data = tf.placeholder(tf.float32, [batch_size, None, input_size], name='input')
        self.target_output = tf.placeholder(tf.float32, [batch_size, None, output_size], name='targets')
        self.sequence_length = tf.placeholder(tf.int32, name='sequence_length')

        self.build_graph()


    def _step_op(self, step, memory_state, controller_state=None):
        last_read_vectors = memory_state[6]
        pre_output, interface, nn_state = None, None, None
        if self.controller.has_recurrent_nn:
            pre_output, interface, nn_state = self.controller.process_input(step, last_read_vectors, controller_state)
        else:
            pre_output, interface = self.controller.process_input(step, last_read_vectors)

        usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector = self.memory.write(
            memory_state[0], memory_state[1], memory_state[5], memory_state[4], memory_state[2], memory_state[3],
            interface['write_key'], interface['write_strength'], interface['free_gates'], interface['allocation_gate'],
            interface['write_gate'], interface['write_vector'], interface['erase_vector'])
        read_weightings, read_vectors = self.memory.read(memory_matrix, memory_state[5], interface['read_keys'], interface['read_strengths'], link_matrix, interface['read_modes'],)

        if self.model == "LSTM":
            return [memory_matrix, usage_vector, precedence_vector, link_matrix, write_weighting, read_weightings, read_vectors, self.controller.final_output(pre_output, read_vectors), interface['free_gates'], interface['allocation_gate'], interface['write_gate'],
                nn_state[0] if nn_state is not None else tf.zeros(1), nn_state[1] if nn_state is not None else tf.zeros(1)]
        elif self.model == "EUNN":
            return [memory_matrix, usage_vector, precedence_vector, link_matrix, write_weighting, read_weightings, read_vectors, self.controller.final_output(pre_output, read_vectors), interface['free_gates'], interface['allocation_gate'], interface['write_gate'],
                nn_state if nn_state is not None else tf.zeros(1)]

    def _loop_body(self, time, memory_state, outputs, free_gates, allocation_gates, write_gates, read_weightings, write_weightings, usage_vectors, controller_state):
        step_input = self.unpacked_input_data.read(time)
        output_list = self._step_op(step_input, memory_state, controller_state)
        new_controller_state = tf.zeros(1)
        new_memory_state = tuple(output_list[0:7])
        if self.model == "LSTM":
            new_controller_state = LSTMStateTuple(output_list[11], output_list[12])
        elif self.model == "EUNN":
            new_controller_state = (output_list[11])
        outputs = outputs.write(time, output_list[7])

        free_gates = free_gates.write(time, output_list[8])
        allocation_gates = allocation_gates.write(time, output_list[9])
        write_gates = write_gates.write(time, output_list[10])
        read_weightings = read_weightings.write(time, output_list[5])
        write_weightings = write_weightings.write(time, output_list[4])
        usage_vectors = usage_vectors.write(time, output_list[1])
        return (time + 1, new_memory_state, outputs, free_gates,allocation_gates, write_gates, read_weightings, write_weightings, usage_vectors, new_controller_state)

    def build_graph(self):
        self.unpacked_input_data = unpack_into_tensorarray(self.input_data, 1, self.sequence_length)
        outputs = tf.TensorArray(tf.float32, self.sequence_length)
        free_gates = tf.TensorArray(tf.float32, self.sequence_length)
        allocation_gates = tf.TensorArray(tf.float32, self.sequence_length)
        write_gates = tf.TensorArray(tf.float32, self.sequence_length)
        read_weightings = tf.TensorArray(tf.float32, self.sequence_length)
        write_weightings = tf.TensorArray(tf.float32, self.sequence_length)
        usage_vectors = tf.TensorArray(tf.float32, self.sequence_length)
        controller_state = self.controller.get_state() if self.controller.has_recurrent_nn else (tf.zeros(1), tf.zeros(1))
        memory_state = self.memory.init_memory()
        if self.model == "LSTM":
            if not isinstance(controller_state, LSTMStateTuple):
                controller_state = LSTMStateTuple(controller_state[0], controller_state[1])
        final_results = None

        with tf.variable_scope("sequence_loop") as scope:
            time = tf.constant(0, dtype=tf.int32)
            final_results = tf.while_loop(
                cond=lambda time, *_: time < self.sequence_length,
                body=self._loop_body,
                loop_vars=(
                    time, memory_state, outputs,
                    free_gates, allocation_gates, write_gates,
                    read_weightings, write_weightings,
                    usage_vectors, controller_state
                ),
                parallel_iterations=32,
                swap_memory=True
            )

        dependencies = []
        if self.controller.has_recurrent_nn:
            dependencies.append(self.controller.update_state(final_results[9]))
        with tf.control_dependencies(dependencies):
            self.packed_output = pack_into_tensor(final_results[2], axis=1)
            self.packed_memory_view = {
                'free_gates': pack_into_tensor(final_results[3], axis=1),
                'allocation_gates': pack_into_tensor(final_results[4], axis=1),
                'write_gates': pack_into_tensor(final_results[5], axis=1),
                'read_weightings': pack_into_tensor(final_results[6], axis=1),
                'write_weightings': pack_into_tensor(final_results[7], axis=1),
                'usage_vectors': pack_into_tensor(final_results[8], axis=1)
            }


    def get_outputs(self):
        return self.packed_output, self.packed_memory_view

    def save(self, session, ckpts_dir, name):
        checkpoint_dir = os.path.join(ckpts_dir, name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        tf.train.Saver(tf.trainable_variables()).save(session, os.path.join(checkpoint_dir, 'model.ckpt'))

    def restore(self, session, ckpts_dir, name):
        tf.train.Saver(tf.trainable_variables()).restore(session, os.path.join(ckpts_dir, name, 'model.ckpt'))
