import tensorflow as tf
import numpy as np
import pickle
import getopt
import sys
import os
import re
import time
from model.recurrent_controller import RecurrentController
from model.dnc.dnc import DNC
from util import *

ckpts_dir = 'checkpoints/'
model = "LSTM"
testData = "mydata"
options,_ = getopt.getopt(sys.argv[1:], '', ['model=', 'test_data='])
for opt in options:
    if opt[0] == '--model':
        model = opt[1]
    elif opt[0] == '--test_data':
        testData = opt[1]
    
llprint("Loading Dictionary ... ")
lexicon_dictionary = load(testData+'/lexicon-dict.pkl')
question_code = lexicon_dictionary["?"]
target_code = lexicon_dictionary["-"]
llprint("Done!\n")
logger = open("run.log","w")

test_files = []
for entryname in os.listdir(testData + '/test/'):
    entry_path = os.path.join(testData + '/test/', entryname)
    if os.path.isfile(entry_path):
        test_files.append(entry_path)
test_files.sort()

tasks_results = {}
tasks_names = {}
for test_file in test_files:
    test_data = load(test_file)
    task_regexp = r'qa([0-9]{1,2})_([a-z\-]*)_test.txt.pkl'
    task_filename = os.path.basename(test_file)
    task_match_obj = re.match(task_regexp, task_filename)
    task_number = int(task_match_obj.group(1))
    task_name = task_match_obj.group(2).replace('-', ' ')
    tasks_names[task_number] = task_name
    counter = 0
    results = []

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:
            ncomputer = DNC(RecurrentController, model, input_size=len(lexicon_dictionary), output_size=len(lexicon_dictionary), max_sequence_length=100, memory_words_num=256, memory_word_size=64, memory_read_heads=4, batch_size=1)
            ncomputer.restore(session, ckpts_dir, model + testData + "_" + str(task_number))
            outputs, _ = ncomputer.get_outputs()
            softmaxed = tf.nn.softmax(outputs)
            llprint("Task %d: %s ... %d/%d" % (task_number, task_name, counter, len(test_data)))
            
            correct = 0
            overall = 0
            for story in test_data:
                astory = np.array(story['inputs'])
                questions_indecies = np.argwhere(astory == question_code)
                questions_indecies = np.reshape(questions_indecies, (-1,))
                target_mask = (astory == target_code)
                desired_answers = np.array(story['outputs'])
                input_vec, _, seq_len, _ = prepare_sample(story, target_code, len(lexicon_dictionary))
                softmax_output = session.run(softmaxed, feed_dict={ncomputer.input_data: input_vec, ncomputer.sequence_length: seq_len})
                softmax_output = np.squeeze(softmax_output, axis=0)
                given_answers = np.argmax(softmax_output[target_mask], axis=1)
                answers_cursor = 0
                for question_indx in questions_indecies:
                    question_grade = []
                    targets_cursor = question_indx + 1
                    while targets_cursor < len(astory) and astory[targets_cursor] == target_code:
                        question_grade.append(given_answers[answers_cursor] == desired_answers[answers_cursor])
                        answers_cursor += 1
                        targets_cursor += 1
                    results.append(np.prod(question_grade))
                    correct += int(np.prod(question_grade))
                    overall += 1
                counter += 1 
                llprint("\rTask %d: %s ... %d/%d" % (task_number, task_name, counter, len(test_data)))
            
            error_rate = 1. - np.mean(results)
            tasks_results[task_number] = error_rate
            print( "\nAccuracy: " + "{:.2f}".format(correct / float(overall) * 100.0) + "% Time elapsed: " + "{:.6}".format(time.time()-timeTaskStart) + " seconds")
            logger.write("{:.2f}".format(correct / float(overall) * 100.0) + " " + "{:.6}".format(time.time()-timeTaskStart) + "\n")

logger.close()
print("\n")
print("%-27s%-27s" % ("Task", "Result"))
print("-------------------------------------------------------------------")
for k in range(len(tasks_results)):
    task_id = str(k + 1)
    task_result = "%.2f%%" % (tasks_results[task_id] * 100)
    print("%-27s%-27s" % (tasks_names[task_id], task_result))
print("-------------------------------------------------------------------")
all_tasks_results = [v for _,v in tasks_results.iteritems()]
results_mean = "%.2f%%" % (np.mean(all_tasks_results) * 100)
failed_count = "%d" % (np.sum(np.array(all_tasks_results) > 0.05))

print("%-27s%-27s" % ("Mean Err.", results_mean))
print("%-27s%-27s" % ("Failed (err. > 5%)", failed_count))
