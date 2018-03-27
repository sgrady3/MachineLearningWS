#!/usr/bin/python3
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
import sys,getopt

#### load the test data and save the data into the format which is suitable in fellow analysis
def read_test_data():
    mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
    X_test = np.stack([np.reshape(img,(28,28)) for img in mnist.test.images])
    y_test = mnist.test.labels
    ####### X_test.npy store the test data(10000 images of numbers)#######
    ####### y_label.npy store the real value of the 10000 numbers  #######
    np.save("./X_test.npy",X_test)
    np.save("./y_label.npy",y_test)

#calculate how many images/numbers are classified wrongly
def sta_accuracy():
    y_true=np.load("./y_label.npy")
    y_class=np.load("./classes.npy")
    y_true_data=np.stack([np.where(x==1)[0][0] for x in y_true])
    c=np.abs(y_true_data-y_class)
    print("There are "+str(np.sum(c))+" numbers which are predicted wrong")
    print("There are "+str(len(y_class))+" numbers in the input data")

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"m:")
    except getopt.GetoptError:
        print('pipeline.py -m /pre-trained/model/location')
        sys.exit(2)
    for opt,arg in opts:
        if opt == "-m":
            model=arg
    read_test_data()
    #### classify #####
    commend_classify="saved_model_cli run --dir "+model+" --tag_set serve --signature_def classify --inputs image=X_test.npy --outdir ./"
    os.system(commend_classify)

    sta_accuracy()

main(sys.argv[1:])
