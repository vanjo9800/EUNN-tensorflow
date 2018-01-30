import tensorflow as tf
import numpy as np
import pickle
import getopt
import time
import sys
import os
import re
from model.dnc.dnc import DNC
from model.recurrent_controller import RecurrentController
from util import *

if __name__ == '__main__':
    model = "LSTM"
    trainData = "en-10k"
    options,_ = getopt.getopt(sys.argv[1:], '', ['model=', 'data='])
    for opt in options:
        if opt[0] == '--model':
            model = opt[1]
        elif opt[0] == '--data':
            trainData = opt[1]
   
    ckpts_dir = '/data/fb-babi-model/checkpoints'
    data_dir = '/data/fb-babi/processed/'
    llprint("Loading Dictionary ... ")
    lexicon_dict = load(os.path.join(data_dir, 'lexicon-dict.pkl'))
    question_code = lexicon_dict["?"]
    target_code = lexicon_dict["-"]
    llprint("Done!\n")
    logger = open("run.log","w")

    batch_size = 1
    input_size = output_size = len(lexicon_dict)
    sequence_max_length = 100
    word_space_size = len(lexicon_dict)
    words_count = 256
    word_size = 64
    read_heads = 4
    learning_rate = 1e-4
    momentum = 0.9
    tasks_results = {} 

    train_files = []
    for entryname in os.listdir(data_dir +'/train/'):
        entry_path = os.path.join(data_dir +'/train/', entryname)
        if os.path.isfile(entry_path):
            train_files.append(entry_path)
    train_files.sort()

    for train_file in train_files:
        task_regexp = r'qa([0-9]{1,2})_([a-z\-]*)_train.txt.pkl'
        task_filename = os.path.basename(train_file)
        task_match_obj = re.match(task_regexp, task_filename)
        task_number = int(task_match_obj.group(1))
        task_name = task_match_obj.group(2).replace('-', ' ')
        
        test_file = train_file.replace("train","test")
        train_data = load(train_file)
        test_data = load(test_file)
        
        print("Assessing on task " + str(task_number)+ ": " + task_name)

        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:
                llprint("Building Computational Graph ... ")
                ncomputer = DNC(RecurrentController, model, input_size, output_size, sequence_max_length, words_count, word_size, read_heads, batch_size)
                output, _ = ncomputer.get_outputs()
                loss_weights = tf.placeholder(tf.float32, [batch_size, None, 1])
                loss = tf.reduce_mean(loss_weights * tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=ncomputer.target_output))
                optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
                gradients = optimizer.compute_gradients(loss)
                for i, (grad, var) in enumerate(gradients):
                    if grad is not None:
                        gradients[i] = (tf.clip_by_value(grad, -10, 10), var)
                apply_gradients = optimizer.apply_gradients(gradients)
                softmaxed = tf.nn.softmax(output)
                llprint("Done!\n")

                llprint("Initializing Variables ... ")
                session.run(tf.global_variables_initializer())
                llprint("Done!\n")

                last_100_losses = []
                start_time_100 = time.time()
                end_time_100 = None
                avg_100_time = 0.
                avg_counter = 0
                counter = 0
                for sample in train_data:
                    try:
                        llprint("\rIteration %d/%d" % (counter, len(train_data)))
                        input_data, target_output, seq_len, weights = prepare_sample(sample, lexicon_dict['-'], word_space_size)
                        loss_value, _ = session.run([loss,apply_gradients], feed_dict={ncomputer.input_data: input_data, ncomputer.target_output: target_output, ncomputer.sequence_length: seq_len, loss_weights: weights})
                        last_100_losses.append(loss_value)
                        logger.write(str(task_number) + " " + str(counter) + " " +"{:.7f}".format(loss_value) + "\n")
                        if counter%100==0:
                            llprint("\n\tAvg. Cross-Entropy: %.7f\n" % (np.mean(last_100_losses)))

                            end_time_100 = time.time()
                            elapsed_time = (end_time_100 - start_time_100) / 60
                            avg_counter += 1
                            avg_100_time += (1. / avg_counter) * (elapsed_time - avg_100_time)
                            estimated_time = (avg_100_time * ((len(train_data) - counter) / 20.)) / 60.
                            llprint("\tAvg. 100 iterations time: %.2f minutes\n" % (avg_100_time))
                            llprint("\tApprox. time to completion: %.2f hours\n" % (estimated_time))
                            start_time_100 = time.time()
                            last_100_losses = []
                        counter += 1
                    except KeyboardInterrupt:
                        llprint("\nSaving Emergency Checkpoint ... "),
                        ncomputer.save(session, ckpts_dir, model + trainData + 'Rec_%d' % (task_number))
                        llprint("Done!\n")
                        sys.exit(0)
                
                counter = 0
                results = []
                correct = 0
                overall = 0
                for story in test_data:
                    astory = np.array(story['inputs'])
                    questions_indecies = np.argwhere(astory == question_code)
                    questions_indecies = np.reshape(questions_indecies, (-1,))
                    target_mask = (astory == target_code)
                    desired_answers = np.array(story['outputs'])
                    input_vec, _, seq_len, _ = prepare_sample(story, target_code, len(lexicon_dict))
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
                print( "\nAccuracy: " + "{:.2f}".format(correct / float(overall) * 100.0))
                logger.write("{:.2f}".format(correct / float(overall) * 100.0) + "\n")
                
                llprint("\nSaving Checkpoint ... "),
                ncomputer.save(session, ckpts_dir, model + trainData + '%d' % (task_number))
                llprint("Done!\n")
    logger.close()
