#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import sys, getopt

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def main(argv):
    inputfile = ''
    outputfile = ''
    labels = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:l:",["input=","output=","labels="])
    except getopt.GetoptError:
        print('plotter.py -i <inputfile> -o <outputfile> -l <labels>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('plotter.py -i <inputfile> -o <outputfile> -l <labels>')
            sys.exit()
        elif opt in ("-i", "--input"):
            inputfile = arg
        elif opt in ("-o", "--output"):
            outputfile = arg
        elif opt in ("-l", "--labels"):
            labels = arg
    inputfile = inputfile.split(',')
    colors = ['b','orange','r','orange']
    cnt = 0
    plt.title("Speech Commands Dataset")
    plt.xlabel("Iterations")
    plt.ylabel("Cross Entropy")
    labels = labels.split(',')
    for data in inputfile:
        data = open(data, 'r')
        next(data)
        iterations = []
        results = []
        for log in data:
            log = log.split(',')
            iterations.append(log[1])
            results.append(log[2])
        iterations = np.array(iterations).astype(np.float)
        results = np.array(results).astype(np.float)
        results_smooth = movingaverage(results,0.2*iterations.size)
        plt.plot(iterations[len(iterations)-len(results_smooth):],results_smooth,colors[cnt],label=labels[cnt])
        cnt = cnt + 1
    plt.legend()
    plt.savefig(outputfile)
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
