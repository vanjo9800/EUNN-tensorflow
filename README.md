# bAbI Tasks Challenge
A test tool for neural networks against the [bAbI tasks](https://research.fb.com/downloads/babi/) challenge developed by Facebook. A particular neural network cell model is integrated into a [Differential Neural Computer](http://www.nature.com/nature/journal/v538/n7626/full/nature20101.html?foxtrotcallback=true) developed by DeepMind and implemented by [Mostafa Samir](https://github.com/Mostafa-Samir/DNC-tensorflow).

# Usage
The test suite can be executed via the **main.sh** run script. It can be used in the following way:
```
./main.sh *data directory* *model* *data name*
```
This performs the three stages of testing (preprocessing the data, training, and testing the network) giving status messages during the process and returning the overall important information at the end.

If you want to execute the three functions of the test suite individually, look below.

# Data preprocessing
First, the data should be preprocessed in order to be suitable for the training and testing processes. This can be accomplished via the following command:
```
python src/preprocess.py --data_dir=*data directory*
```

The given data directory should contain two subdirectories: a *train/* directory containing the training data samples and a *test/* directory containing the testing data samples. The preprocessing script generates a dictionary file and a single encoded file for each train and test data sample. The processes data can be found in the *data/* directory.

# Training procedure
The second stage of operation of the test suite is the training procedure. It can be accomplished via the following command:
```
python src/train.py --model=*model* --train_data=*training dataset*
```

The process reads the previously generated dictionary and encoded training data and runs the network over them. The program reports a status message on every 100 iterations containg the average loss, elapsed time, and estimated remaining time. In the end, a checkpoint with the trained model is generated under directory *checkpoints/*. If the training procedure is unexpectedly stopped, it creates an emergency checkpoint with *Rec* in its name with the learning model to the current iterations.

# Testing procedure
The third stage of operation of the test suite is the testing procedure of the previously trained model. It can be accomplished via the following command:
```
python src/test.py --model=*model* --test_data=*testing dataset*
```

The process reads the previously generated dictionary, encoded testing data, and trained network model from a checkpoint and runs it over the testing data. The program reports a status message after every task reporting the accuracy. In addition, after each test sample a counter is increased showing the remaining test samples. In the end, a table with the accuracies on all tasks is provided comparing them with the maximum achieved accuracy by Mostafa Samir's implementation.

# Web visualizer
A work on a web visualizer has begun in order to present the results of the network model in a more convenient way. The files associated with this addition are located under *web/* and *src/webTester.py*. 

# Future development
Useful additions to the testing suite will be:
- Web visualizer implementation
