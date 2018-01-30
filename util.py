import os
import pickle
import sys
import re
import numpy as np

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def load(path):
    return pickle.load(open(path, 'rb'))

def parse_data(data_dir):
    stories = []
    for entryname in os.listdir(data_dir+'/'):
        entry_path = os.path.join(data_dir+'/', entryname)
        if os.path.isfile(entry_path):
            test_data = load(entry_path)
            task_regexp = r'qa([0-9]{1,2})_([a-z\-]*)_test.txt.pkl'
            task_filename = os.path.basename(entry_path)
            task_match_obj = re.match(task_regexp, task_filename)
            task_number = task_match_obj.group(1)
            task_name = task_match_obj.group(2).replace('-', ' ')
            tasks_names[task_number] = task_name
            stories.append((entry_path,len(test_data)))

    return stories

def onehot(index, size):
    vec = np.zeros(size, dtype=np.float32)
    vec[int(index)] = 1.0
    return vec

def prepare_sample(sample, target_code, word_space_size):
    input_vec = np.array(sample['inputs'], dtype=np.float32)
    output_vec = np.array(sample['inputs'], dtype=np.float32)
    seq_len = input_vec.shape[0]
    weights_vec = np.zeros(seq_len, dtype=np.float32)
    target_mask = (input_vec == target_code)
    output_vec[target_mask] = sample['outputs']
    weights_vec[target_mask] = 1.0
    input_vec = np.array([onehot(code, word_space_size) for code in input_vec])
    output_vec = np.array([onehot(code, word_space_size) for code in output_vec])

    return (
        np.reshape(input_vec, (1, -1, word_space_size)),
        np.reshape(output_vec, (1, -1, word_space_size)),
        seq_len,
        np.reshape(weights_vec, (1, -1, 1))
    )
