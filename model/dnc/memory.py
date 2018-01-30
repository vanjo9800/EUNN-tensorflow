import tensorflow as tf
import numpy as np
from .utility import *

class Memory:
    def __init__(self, words_num=256, word_size=64, read_heads=4, batch_size=1):
        self.words_num = words_num
        self.word_size = word_size
        self.read_heads = read_heads
        self.batch_size = batch_size
        self.I = tf.constant(np.identity(words_num, dtype=np.float32))
        self.index_mapper = tf.constant(
            np.cumsum([0] + [words_num] * (batch_size - 1), dtype=np.int32)[:, np.newaxis]
        )

    def init_memory(self):
        return (
            tf.fill([self.batch_size, self.words_num, self.word_size], 1e-6),  # initial memory matrix
            tf.zeros([self.batch_size, self.words_num, ]),  # initial usage vector
            tf.zeros([self.batch_size, self.words_num, ]),  # initial precedence vector
            tf.zeros([self.batch_size, self.words_num, self.words_num]),  # initial link matrix
            tf.fill([self.batch_size, self.words_num, ], 1e-6),  # initial write weighting
            tf.fill([self.batch_size, self.words_num, self.read_heads], 1e-6),  # initial read weightings
            tf.fill([self.batch_size, self.word_size, self.read_heads], 1e-6),  # initial read vectors
        )

    def get_lookup_weighting(self, memory_matrix, keys, strengths):
        normalized_memory = tf.nn.l2_normalize(memory_matrix, 2)
        normalized_keys = tf.nn.l2_normalize(keys, 1)
        similiarity = tf.matmul(normalized_memory, normalized_keys)
        strengths = tf.expand_dims(strengths, 1)
        return tf.nn.softmax(similiarity * strengths, 1)


    def update_usage_vector(self, usage_vector, read_weightings, write_weighting, free_gates):
        free_gates = tf.expand_dims(free_gates, 1)
        retention_vector = tf.reduce_prod(1 - read_weightings * free_gates, 2)
        updated_usage = (usage_vector + write_weighting - usage_vector * write_weighting)  * retention_vector
        return updated_usage


    def get_allocation_weighting(self, sorted_usage, free_list):
        shifted_cumprod = tf.cumprod(sorted_usage, axis = 1, exclusive=True)
        unordered_allocation_weighting = (1 - sorted_usage) * shifted_cumprod
        mapped_free_list = free_list + self.index_mapper
        flat_unordered_allocation_weighting = tf.reshape(unordered_allocation_weighting, (-1,))
        flat_mapped_free_list = tf.reshape(mapped_free_list, (-1,))
        flat_container = tf.TensorArray(tf.float32, self.batch_size * self.words_num)
        flat_ordered_weightings = flat_container.scatter(flat_mapped_free_list, flat_unordered_allocation_weighting)
        packed_wightings = flat_ordered_weightings.stack()
        return tf.reshape(packed_wightings, (self.batch_size, self.words_num))


    def update_write_weighting(self, lookup_weighting, allocation_weighting, write_gate, allocation_gate):
        lookup_weighting = tf.squeeze(lookup_weighting)
        updated_write_weighting = write_gate * (allocation_gate * allocation_weighting + (1 - allocation_gate) * lookup_weighting)
        return updated_write_weighting


    def update_memory(self, memory_matrix, write_weighting, write_vector, erase_vector):
        write_weighting = tf.expand_dims(write_weighting, 2)
        write_vector = tf.expand_dims(write_vector, 1)
        erase_vector = tf.expand_dims(erase_vector, 1)
        erasing = memory_matrix * (1 - tf.matmul(write_weighting, erase_vector))
        writing = tf.matmul(write_weighting, write_vector)
        updated_memory = erasing + writing
        return updated_memory


    def update_precedence_vector(self, precedence_vector, write_weighting):
        reset_factor = 1 - tf.reduce_sum(write_weighting, 1, keep_dims=True)
        updated_precedence_vector = reset_factor * precedence_vector + write_weighting
        return updated_precedence_vector


    def update_link_matrix(self, precedence_vector, link_matrix, write_weighting):
        write_weighting = tf.expand_dims(write_weighting, 2)
        precedence_vector = tf.expand_dims(precedence_vector, 1)
        reset_factor = 1 - pairwise_add(write_weighting, is_batch=True)
        updated_link_matrix = reset_factor * link_matrix + tf.matmul(write_weighting, precedence_vector)
        updated_link_matrix = (1 - self.I) * updated_link_matrix  # eliminates self-links
        return updated_link_matrix


    def get_directional_weightings(self, read_weightings, link_matrix):
        forward_weighting = tf.matmul(link_matrix, read_weightings)
        backward_weighting = tf.matmul(link_matrix, read_weightings, adjoint_a=True)
        return forward_weighting, backward_weighting


    def update_read_weightings(self, lookup_weightings, forward_weighting, backward_weighting, read_mode):
        backward_mode = tf.expand_dims(read_mode[:, 0, :], 1) * backward_weighting
        lookup_mode = tf.expand_dims(read_mode[:, 1, :], 1) * lookup_weightings
        forward_mode = tf.expand_dims(read_mode[:, 2, :], 1) * forward_weighting
        updated_read_weightings = backward_mode + lookup_mode + forward_mode
        return updated_read_weightings


    def update_read_vectors(self, memory_matrix, read_weightings):
        updated_read_vectors = tf.matmul(memory_matrix, read_weightings, adjoint_a=True)
        return updated_read_vectors


    def write(self, memory_matrix, usage_vector, read_weightings, write_weighting, precedence_vector, link_matrix,  key, strength, free_gates, allocation_gate, write_gate, write_vector, erase_vector):
        lookup_weighting = self.get_lookup_weighting(memory_matrix, key, strength)
        new_usage_vector = self.update_usage_vector(usage_vector, read_weightings, write_weighting, free_gates)
        sorted_usage, free_list = tf.nn.top_k(-1 * new_usage_vector, self.words_num)
        sorted_usage = -1 * sorted_usage
        allocation_weighting = self.get_allocation_weighting(sorted_usage, free_list)
        new_write_weighting = self.update_write_weighting(lookup_weighting, allocation_weighting, write_gate, allocation_gate)
        new_memory_matrix = self.update_memory(memory_matrix, new_write_weighting, write_vector, erase_vector)
        new_link_matrix = self.update_link_matrix(precedence_vector, link_matrix, new_write_weighting)
        new_precedence_vector = self.update_precedence_vector(precedence_vector, new_write_weighting)
        return new_usage_vector, new_write_weighting, new_memory_matrix, new_link_matrix, new_precedence_vector


    def read(self, memory_matrix, read_weightings, keys, strengths, link_matrix, read_modes):
        lookup_weighting = self.get_lookup_weighting(memory_matrix, keys, strengths)
        forward_weighting, backward_weighting = self.get_directional_weightings(read_weightings, link_matrix)
        new_read_weightings = self.update_read_weightings(lookup_weighting, forward_weighting, backward_weighting, read_modes)
        new_read_vectors = self.update_read_vectors(memory_matrix, new_read_weightings)
        return new_read_weightings, new_read_vectors
