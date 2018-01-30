import sys
import pickle
import getopt
import numpy as np
from shutil import rmtree
from os import listdir, mkdir
from os.path import join, isfile, isdir, dirname, basename, normpath, abspath, exists

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def create_dictionary(files_list):
    lexicons_dict = {}
    id_counter = 0

    llprint("Creating Dictionary ... 0/%d" % (len(files_list)))
    for indx, filename in enumerate(files_list):
        with open(filename, 'r') as fobj:
            for line in fobj:
                line = line.replace('.', ' .')
                line = line.replace('?', ' ?')
                line = line.replace(',', ' ')

                for word in line.split():
                    if not word.lower() in lexicons_dict and word.isalpha():
                        lexicons_dict[word.lower()] = id_counter
                        id_counter += 1

        llprint("\rCreating Dictionary ... %d/%d" % ((indx + 1), len(files_list)))
    print("\rCreating Dictionary ... Done!")
    return lexicons_dict


def encode_data(files_list, lexicons_dictionary):
    files = {}
    story_inputs = None
    story_outputs = None
    stories_lengths = []
    answers_flag = False

    llprint("Encoding Data ... 0/%d" % (len(files_list)))
    for indx, filename in enumerate(files_list):
        files[filename] = []
        with open(filename, 'r') as fobj:
            time = 0
            for line in fobj:
                time += 1
                line = line.replace('.', ' .')
                line = line.replace('?', ' ?')
                line = line.replace(',', ' ')
                answers_flag = False

                for i, word in enumerate(line.split()):
                    if word == '1' and i == 0:
                        time = 1
                        if not story_inputs is None:
                            stories_lengths.append(len(story_inputs))
                            files[filename].append({'inputs':story_inputs, 'outputs': story_outputs})
                        story_inputs = []
                        story_outputs = []
                    
                    if word.isalpha() or word == '?' or word == '.':
                        if not answers_flag:
                            story_inputs.append(lexicons_dictionary[word.lower()])
                        else:
                            story_inputs.append(lexicons_dictionary['-'])
                            story_outputs.append(lexicons_dictionary[word.lower()])

                        if not answers_flag:
                            answers_flag = (word == '?')

        llprint("\rEncoding Data ... %d/%d" % (indx + 1, len(files_list)))

    print("\rEncoding Data ... Done!")
    return files, stories_lengths


if __name__ == '__main__':
    task_dir = dirname(abspath(__file__))
    options,_ = getopt.getopt(sys.argv[1:], '', ['data_dir=', 'single_train'])
    data_dir = None
    joint_train = False
    for opt in options:
        if opt[0] == '--data_dir':
            data_dir = opt[1]
        if opt[0] == '--single_train':
            joint_train = False
    if data_dir is None:
        raise ValueError("data_dir argument cannot be None")
    files_list = []
    if not exists(join(task_dir, '../data')):
        mkdir(join(task_dir, '../data'))
    processed_data_dir = join(task_dir, '../data', basename(normpath(data_dir)))
    if exists(processed_data_dir) and isdir(processed_data_dir):
        cont = input("The dataset has been previously processed. Process again? [y/N]")
        if cont != 'y': sys.exit()
        rmtree(processed_data_dir)

    for entryname in listdir(data_dir):
        entry_path = join(data_dir, entryname)
        if isfile(entry_path):
            files_list.append(entry_path)
    lexicon_dictionary = create_dictionary(files_list)
    lexicon_count = len(lexicon_dictionary)
    lexicon_dictionary['?'] = lexicon_count
    lexicon_dictionary['.'] = lexicon_count + 1
    lexicon_dictionary['-'] = lexicon_count + 2
    encoded_files, stories_lengths = encode_data(files_list, lexicon_dictionary)
    stories_lengths = np.array(stories_lengths)
    print("Total Number of stories: %d" % (len(stories_lengths)))

    train_data_dir = join(processed_data_dir, 'train')
    test_data_dir = join(processed_data_dir, 'test')
    if exists(processed_data_dir) and isdir(processed_data_dir):
        rmtree(processed_data_dir)

    mkdir(processed_data_dir)
    mkdir(train_data_dir)
    mkdir(test_data_dir)

    llprint("Saving processed data to disk ... ")
    pickle.dump(lexicon_dictionary, open(join(processed_data_dir, 'lexicon-dict.pkl'), 'wb'))

    joint_train_data = []
    for filename in encoded_files:
        if filename.endswith("test.txt"):
            pickle.dump(encoded_files[filename], open(join(test_data_dir, basename(filename) + '.pkl'), 'wb'))
        elif filename.endswith("train.txt"):
            if not joint_train:
                pickle.dump(encoded_files[filename], open(join(train_data_dir, basename(filename) + '.pkl'), 'wb'))
            else:
                joint_train_data.extend(encoded_files[filename])

    if joint_train:
        pickle.dump(joint_train_data, open(join(train_data_dir, 'train.pkl'), 'wb'))

    llprint("Done!\n")
