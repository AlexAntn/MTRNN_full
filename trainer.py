from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from CTRNN import CTRNNModel

import time
import operator
import io
import array
import datetime

import os
import sys

import itertools


import pandas as pd
from sklearn.decomposition import PCA

def processWords(sentence, x_train, k):
    sentence = sentence.replace("\n", "").replace("\r", "")
    print(sentence)
    for f in range(0, len(sentence), 1):
        if sentence[f] == ' ' and f <= 29:
            x_train[k, f + 4,26] = 1.0
        elif sentence[f] == '.' and f <= 29:
            x_train[k, f + 4,27] = 1.0
        elif f <= 29:
            print(sentence[f])
            x_train[k, f + 4,ord(sentence[f]) - 97] = 1.0 

    return x_train 


def get_sentence(verb, obj):
    verb = float(verb)
    obj = float(obj)
    if verb >= 0.0 and verb < 0.1:
        sentence = "slide left"
    elif verb >= 0.1 and verb < 0.2:
        sentence = "slide right"
    elif verb >= 0.2 and verb < 0.3:
        sentence = "touch"
    elif verb >= 0.3 and verb < 0.4:
        sentence = "reach"
    elif verb >= 0.4 and verb < 0.5:
        sentence = "push"
    elif verb >= 0.5 and verb < 0.6:
        sentence = "pull"
    elif verb >= 0.6 and verb < 0.7:
        sentence = "point"
    elif verb >= 0.7 and verb < 0.8:
        sentence = "grasp"
    else:
        sentence = "lift"
    if obj >= 0.0 and obj < 0.1:
        sentence = sentence + " the " + "tractor"
    elif obj >= 0.1 and obj < 0.2:
        sentence = sentence + " the " + "hammer"
    elif obj >= 0.2 and obj < 0.3:
        sentence = sentence + " the " + "ball"
    elif obj >= 0.3 and obj < 0.4:
        sentence = sentence + " the " + "bus"
    elif obj >= 0.4 and obj < 0.5:
        sentence = sentence + " the " + "modi"
    elif obj >= 0.5 and obj < 0.6:
        sentence = sentence + " the " + "car"
    elif obj >= 0.6 and obj < 0.7:
        sentence = sentence + " the " + "cup"
    elif obj >= 0.7 and obj < 0.8:
        sentence = sentence + " the " + "cubes"
    else:
        sentence = sentence + " the " + "spiky"
    sentence = sentence + "."
    return sentence

def get_combination(verb, obj, control_input):
    new_control = np.zeros((1, control_input.shape[1], control_input.shape[2]))
    verb = float(verb)
    obj = float(obj)
    if verb >= 0.0 and verb < 0.1:
        new_control[0, :, 0:4] = [0.0, 0.0, 0.0, 1.0]
    elif verb >= 0.1 and verb < 0.2:
        new_control[0, :, 0:4] = [0.0, 0.0, 1.0, 0.0]
    elif verb >= 0.2 and verb < 0.3:
        new_control[0, :, 0:4] = [0.0, 0.0, 1.0, 1.0]
    elif verb >= 0.3 and verb < 0.4:
        new_control[0, :, 0:4] = [0.0, 1.0, 0.0, 0.0]
    elif verb >= 0.4 and verb < 0.5:
        new_control[0, :, 0:4] = [0.0, 1.0, 0.0, 1.0]
    elif verb >= 0.5 and verb < 0.6:
        new_control[0, :, 0:4] = [0.0, 1.0, 1.0, 0.0]
    elif verb >= 0.6 and verb < 0.7:
        new_control[0, :, 0:4] = [0.0, 1.0, 1.0, 1.0]
    elif verb >= 0.7 and verb < 0.8:
        new_control[0, :, 0:4] = [1.0, 0.0, 0.0, 0.0]
    else:
        new_control[0, :, 0:4] = [1.0, 0.0, 0.0, 1.0]
    if obj >= 0.0 and obj < 0.1:
        new_control[0, :, 4:8] = [0.0, 0.0, 0.0, 1.0]
    elif obj >= 0.1 and obj < 0.2:
        new_control[0, :, 4:8] = [0.0, 0.0, 1.0, 0.0]
    elif obj >= 0.2 and obj < 0.3:
        new_control[0, :, 4:8] = [0.0, 0.0, 1.0, 1.0]
    elif obj >= 0.3 and obj < 0.4:
        new_control[0, :, 4:8] = [0.0, 1.0, 0.0, 0.0]
    elif obj >= 0.4 and obj < 0.5:
        new_control[0, :, 4:8] = [0.0, 1.0, 0.0, 1.0]
    elif obj >= 0.5 and obj < 0.6:
        new_control[0, :, 4:8] = [0.0, 1.0, 1.0, 0.0]
    elif obj >= 0.6 and obj < 0.7:
        new_control[0, :, 4:8] = [0.0, 1.0, 1.0, 1.0]
    elif obj >= 0.7 and obj < 0.8:
        new_control[0, :, 4:8] = [1.0, 0.0, 0.0, 0.0]
    else:
        new_control[0, :, 4:8] = [1.0, 0.0, 0.0, 1.0]

    return new_control

#construction of control sequence (fixed combinations, 6 neurons, activation can be 0, 0.5 or 1.0)
def get_sentence2(verb, obj): # maybe not used##################################
    sentence = ""
    if verb == [0.0, 0.0, 0.0, 0.0]:
            verb_string = ""
    elif verb == [0.0, 0.0, 0.0, 1.0]:
            verb_string = "slide left"
    elif verb == [0.0, 0.0, 1.0, 0.0]:
            verb_string = "slide right"
    elif verb == [0.0, 0.0, 1.0, 1.0]:
            verb_string = "touch"
    elif verb == [0.0, 1.0, 0.0, 0.0]:
            verb_string = "reach"
    elif verb == [0.0, 1.0, 0.0, 1.0]:
            verb_string = "push"
    elif verb == [0.0, 1.0, 1.0, 0.0]:
            verb_string = "pull"
    elif verb == [0.0, 1.0, 1.0, 1.0]:
            verb_string = "point at"
    elif verb == [1.0, 0.0, 0.0, 0.0]:
            verb_string = "grasp"
    elif verb == [1.0, 0.0, 0.0, 1.0]:
            verb_string = "lift"
    if obj == [0.0, 0.0, 0.0, 0.0]:
            obj_string = ""
    elif obj == [0.0, 0.0, 0.0, 1.0]:
            obj_string = "tractor"
    elif obj == [0.0, 0.0, 1.0, 0.0]:
            obj_string = "hammer"
    elif obj == [0.0, 0.0, 1.0, 1.0]:
            obj_string = "ball"
    elif obj == [0.0, 1.0, 0.0, 0.0]:
            obj_string = "bus"
    elif obj == [0.0, 1.0, 0.0, 1.0]:
            obj_string = "modi"
    elif obj == [0.0, 1.0, 1.0, 0.0]:
            obj_string = "car"
    elif obj == [0.0, 1.0, 1.0, 1.0]:
            obj_string = "cup"
    elif obj == [1.0, 0.0, 0.0, 0.0]:
            obj_string = "cubes"
    elif obj == [1.0, 0.0, 0.0, 1.0]:
            obj_string = "spiky"
    if obj_string != "" and verb_string != "":
        sentence = verb_string + " the " + obj_string + "."
    else:
        sentence = verb_string + obj_string +"."
    return sentence

######################################################################################
# This function loads data from a file, to train the network
# inputs are sequential (and always same order). 
def loadTrainingData(LangInputNeurons, MotorInputNeurons, Lang_stepEachSeq, Motor_stepEachSeq, numSeq):

    stepEachSeq = Lang_stepEachSeq + Motor_stepEachSeq

    # sequence of letters
    x_train = np.asarray(np.zeros((numSeq , stepEachSeq, LangInputNeurons)),dtype=np.float32)
    y_train = 26 * np.asarray(np.ones((numSeq , stepEachSeq)),dtype=np.int32)

    # motor sequence
    m_train = np.asarray(np.zeros((numSeq, stepEachSeq, MotorInputNeurons)), dtype=np.float32)
    m_gener = np.asarray(np.zeros((numSeq, stepEachSeq, MotorInputNeurons)), dtype=np.float32)

    print("steps: ", stepEachSeq)
    print("number of sequences: ", numSeq)

    dataFile = open("mtrnnTD.txt", 'r')
    RANDOM_SEQUENCES = False

    totalSeq = 432
    sequences = []
    if RANDOM_SEQUENCES:
        for i in range(numSeq):
            sequences += [np.random.randint(0, totalSeq)]
            print(sequences[-1])
    else:
        sequences = np.arange(totalSeq)

    sentence_list = []

    sequences = [k for k in range(0, totalSeq, 1)]
    #sequences = [12]#, 65]


    print(sequences)

    k = 0 #number of sequences
    t = -1 #number of saved sequences
    while True:
        line = dataFile.readline()
        if line == "":
            break
        if line.find("SEQUENCE") != -1:
            if k in sequences: # to select random sentences
                #print "found sequence"
                t+=1
                for i in range(0, Motor_stepEachSeq):
                    line = dataFile.readline()
                    line_data = line.split("\t")
                    line_data[-1] = line_data[-1].replace("\r\n",'')
                    if i == 0:
                        sentence = get_sentence(line_data[0], line_data[1])
                        sentence_list += [sentence]
                        #print("sentence: ", sentence)
                        #raw_input()
                        l = 0
                        p = 0
                        for g in range(Lang_stepEachSeq):
                            if l == 4 and p < len(sentence):
                                l = 0
                            #if l == 0:
                                lett = sentence[p]
                                p += 1
                            m_gener[t, g, 0:MotorInputNeurons] = line_data[1:MotorInputNeurons+1]
                            if g < len(sentence)*4+4 and g >=4:
                                if lett == ' ':
                                    x_train[t, g,26] = 1
                                    y_train[t, Motor_stepEachSeq + g] = 26
                                elif lett == '.':
                                    x_train[t, g,27] = 1
                                    y_train[t, Motor_stepEachSeq + g] = 27
                                else:
                                    x_train[t, g, ord(lett) - 97] = 1
                                    y_train[t, Motor_stepEachSeq + g] =  ord(lett) - 97
                            else:
                                x_train[t, g,26] = 1
                                y_train[t, Motor_stepEachSeq + g] = 26
                            l += 1
                    # we save the values for the encoders at each step
                    m_train[t, i,0:MotorInputNeurons] = line_data[1:MotorInputNeurons+1]
                    m_gener[t, i+Lang_stepEachSeq, 0:MotorInputNeurons] = line_data[1:MotorInputNeurons+1]
                    y_train[t, i] = 26
                    x_train[t, Lang_stepEachSeq + i, 26] = 1

                # now we set the motor output to be constant in the end 
                for i in range(Motor_stepEachSeq, stepEachSeq):
                    m_train[t, i,0:MotorInputNeurons] = line_data[1:MotorInputNeurons+1]
                #print("lang: ", y_train[t,:])
                #raw_input()
            # indicator of how many sequences we have gone through
                #plt.plot(m_train[t, :, 16], 'r')
                #plt.plot(m_gener[t, :, 16], 'b')
                #plt.plot(x_train[i, :, 2], 'g')
                #plt.plot(x_train[i, :, 3], 'c')
                #plt.plot(x_train[i, :, 4], 'y')
            plt.show()
            k = k+1 
        if k == totalSeq:
            break
    #m_train[0,0:100, :] = np.ones([1, 100, 42])
    #for i in range(100):
        #m_train[0,i,:] = m_train[0,i,:]/(i+1)
        #m_train[0,i,:] = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    #m_gener[0, 30:130, :] = np.ones([1, 100, 42])
    #for i in range(30,130):
        #m_gener[0,i,:] = m_gener[0,i,:]/(i-29)
        #m_gener[0, i, :] = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
        
    dataFile.close()

    return x_train, y_train, m_train, m_gener, sentence_list

def execute_pca(sentences, States, lang_input, input_layer, lang_dim1, lang_dim2, control_dim, direction, seqsTested, b):

    Mat_S0 = States[:, :,0, 0:lang_input]
    Mat_S1 = States[:, :,1, 0:input_layer]
    Mat_S2 = States[:, :,2, 0:lang_dim1]
    Mat_S3 = States[:, :,3, 0:lang_dim2]
    Mat_S4 = States[:, :,4, 0:control_dim]

#############################################

    component_1 = 0.0
    component_2 = 0.0
    #plt.ion()
    color = 0
    plt.subplot(111)
    plt.subplots_adjust(right = 0.75)
    for i in range(0,seqsTested,1):
        inputdata = pd.DataFrame(data = Mat_S1[i])

        color = i/seqsTested
        color_inv = 1 - color
        position = i/seqsTested
        #print(color)
        pca = PCA(n_components = 2)
        plotdata = pca.fit(Mat_S1[i]).transform(Mat_S1[i])
        #print("data explained by PCA for IO: ", pca.explained_variance_ratio_)
        component_1 += pca.explained_variance_ratio_[0]
        component_2 += pca.explained_variance_ratio_[1]

        for t in range(len(plotdata)):
            plt.scatter(plotdata[t,0], plotdata[t, 1], c=(0.0, 0.0, t/len(plotdata)), marker = 'o')
        plt.plot(plotdata[:,0], plotdata[:, 1], c=(color_inv, color, 0.0))
        #plt.text(plotdata[i,0], plotdata[i, 1], sentences[i+b], c=(color_inv, color, 0.0))
        #plt.text(0.05, position, sentences[i+b], verticalalignment='bottom', horizontalalignment='left', color=(color_inv, color, 0.0))    
        plt.text(1.05, position, sentences[i+b], verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes, color=(color_inv, color, 0.0))

    my_path= os.path.dirname(__file__)
    figure_path = os.path.join(my_path, "figuresIO/")
    plt.title("IO trajectory - "+sentences[b])
    plt.xlabel("PC1 :" + str(component_1/i))
    plt.ylabel("PC2 :" + str(component_2/i))
    plt.grid()

    if direction:
        plt.savefig(figure_path + sentences[b] + '_IO_layer_CS_to_sentences.png', dpi=125)
    else:
        plt.savefig(figure_path + sentences[b] + '_IO_layer_sentences_to_CS.png', dpi=125)
    plt.close()
    #plt.ioff()

#############################

    component_1 = 0.0
    component_2 = 0.0
    color = 0
    plt.subplot(111)
    plt.subplots_adjust(right = 0.75)
    for i in range(0,seqsTested,1):
        inputdata = pd.DataFrame(data = Mat_S2[i])

        color = i/seqsTested
        color_inv = 1 - color
        position = i/seqsTested
        pca = PCA(n_components = 2)
        plotdata = pca.fit(Mat_S2[i]).transform(Mat_S2[i])
        #print("data explained by PCA for IO: ", pca.explained_variance_ratio_)
        component_1 += pca.explained_variance_ratio_[0]
        component_2 += pca.explained_variance_ratio_[1]

        for t in range(len(plotdata)):
            plt.scatter(plotdata[t,0], plotdata[t, 1], c=(0.0, 0.0, t/len(plotdata)), marker = 'o')
        plt.plot(plotdata[:,0], plotdata[:, 1], c=(color_inv, color, 0.0))
        #plt.text(plotdata[i,0], plotdata[i, 1], sentences[i+b])
        plt.text(1.05, position, sentences[i+b], verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes, color=(color_inv, color, 0.0))

    my_path= os.path.dirname(__file__)
    figure_path = os.path.join(my_path, "figuresFC/")
    plt.title("FC trajectory - "+sentences[b]);
    plt.xlabel("PC1 :" + str(component_1/i));
    plt.ylabel("PC2 :" + str(component_2/i));
    plt.grid();

    if direction:
        plt.savefig(figure_path+ sentences[b] + '_FC_layer_CS_to_sentences.png', dpi=125)
    else:
        plt.savefig(figure_path+ sentences[b] + '_FC_layer_sentences_to_CS.png', dpi=125)
    plt.close()

#############################

    component_1 = 0.0
    component_2 = 0.0
    color = 0
    plt.subplot(111)
    plt.subplots_adjust(right = 0.75)
    for i in range(0,seqsTested,1):
        inputdata = pd.DataFrame(data = Mat_S3[i])

        color = i/seqsTested
        color_inv = 1 - color
        position = i/seqsTested
        pca = PCA(n_components = 2)
        plotdata = pca.fit(Mat_S3[i]).transform(Mat_S3[i])
        #print("data explained by PCA for IO: ", pca.explained_variance_ratio_)
        component_1 += pca.explained_variance_ratio_[0]
        component_2 += pca.explained_variance_ratio_[1]

        for t in range(len(plotdata)):
            plt.scatter(plotdata[t,0], plotdata[t, 1], c=(0.0, 0.0, t/len(plotdata)), marker = 'o')
        plt.plot(plotdata[:,0], plotdata[:, 1], color=(color_inv, color, 0.0))
        #plt.text(plotdata[i,0], plotdata[i, 1], sentences[i+b])#, verticalalignment = 'bottom', horizontalalignment = 'left')
        plt.text(1.05, position, sentences[i+b], verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes, color=(color_inv, color, 0.0))

    my_path= os.path.dirname(__file__)
    figure_path = os.path.join(my_path, "figuresSC/")
    plt.title("SC trajectory - "+sentences[b]);
    plt.xlabel("PC1 :" + str(component_1/i));
    plt.ylabel("PC2 :" + str(component_2/i));
    plt.grid();

    if direction:
        plt.savefig(figure_path + sentences[b] + '_SC_layer_CS_to_sentences.png', dpi=125)
    else:
        plt.savefig(figure_path + sentences[b] + '_SC_layer_sentences_to_CS.png', dpi=125)
    plt.close()

###########################################


def plot(loss_list, fig, ax):
    ax.semilogy(loss_list, 'b')
    fig.canvas.flush_events()



########################################## Control Variables ################################
direction = True
alternate = False
alpha = 1
RUN_PCA = False
NEPOCH = 250000 # number of times to train each sentence
threshold_lang = 0.005
threshold_motor = 0.03
average_loss = 1000.0
best_loss = 5
best_loss_lang = 0.5
best_loss_motor = 15#1000.0

loss_list = []
lang_loss_list = [5.0] # just a value so it doesn't stop saving because of this
motor_loss_list = [15.0]

my_path= os.getcwd()

jumps = 1

########################################## Model parameters ################################
lang_input = 28 # size of output/input sentence
input_layer = 40 # IO layer
lang_dim1 = 160 # fast context
lang_dim2 = 35 # slow context (without control neurons)
meaning_dim = 25
motor_dim2 = 35
motor_dim1 = 160
motor_layer = 140
motor_input = 42


numSeq = 432
Lang_stepEachSeq = 100
Motor_stepEachSeq = 100
stepEachSeq = Lang_stepEachSeq + Motor_stepEachSeq

LEARNING_RATE = 5 * 1e-3

MTRNN = CTRNNModel([input_layer, lang_dim1, lang_dim2, meaning_dim, motor_dim2, motor_dim1, motor_layer], [2, 5, 60, 100, 60, 5, 2], stepEachSeq, lang_input, motor_input, LEARNING_RATE)


#################################### acquire data ##########################################
x_train, y_train, m_train, m_gener, sentence_list = loadTrainingData(lang_input, motor_input, Lang_stepEachSeq, Motor_stepEachSeq, numSeq)

old_x = x_train
old_y = y_train
old_m_train = m_train
old_m_gener = m_gener


########## Roll the outputs, so it tries predicting the future #############
m_output = np.zeros([numSeq, stepEachSeq, motor_input], dtype=np.float32)
m_output[:,:,:] = np.roll(m_gener, -1, axis=1)[:,:,0:motor_input]
m_output[:,-1,:] = m_output[:,-2,:]

old_m_output = m_output
old_sentence = sentence_list
old_numSeq = numSeq

################ This needs to be changed later ##################################
exclude_sentences = False
if exclude_sentences:
    numSeq = 431
    print(x_train.shape)
    new_x_train = np.zeros((80, x_train.shape[1], x_train.shape[2]))
    test_x = np.zeros((1, x_train.shape[1], x_train.shape[2]))
    new_x_train[:65] = x_train[:65]
    new_x_train[65:80] = x_train[66:81]
    test_x[0] = x_train[65]
    test_sentence = sentence_list[65]
    print(test_sentence)
    x_train = new_x_train
    new_y_train = np.zeros((80, y_train.shape[1]))
    test_y = np.zeros((1, y_train.shape[1]))
    new_y_train[:65] = y_train[:65]
    new_y_train[65:80] = y_train[66:81]
    test_y[0] = y_train[65]
    y_train = new_y_train
    new_control_input = np.zeros((80, control_input.shape[1], control_input.shape[2]))
    test_control = np.zeros((1, control_input.shape[1], control_input.shape[2]))
    new_control_input[:65] = control_input[:65]
    new_control_input[65:80] = control_input[66:81]
    test_control[0] = control_input[65]
    control_input = new_control_input
    raw_input()
#################################################################################

init_state_IO_l = np.zeros([numSeq, input_layer], dtype = np.float32)
init_state_fc_l = np.zeros([numSeq, lang_dim1], dtype = np.float32)
init_state_sc_l = np.zeros([numSeq, lang_dim2], dtype = np.float32)
init_state_ml = np.zeros([numSeq, meaning_dim], dtype = np.float32)
init_state_IO_m = np.zeros([numSeq, motor_layer], dtype = np.float32)
init_state_fc_m = np.zeros([numSeq, motor_dim1], dtype = np.float32)
init_state_sc_m = np.zeros([numSeq, motor_dim2], dtype = np.float32)

#gate_motor_to_meaning = np.zeros([numSeq, stepEachSeq], dtype = np.float32)
#gate_motor_to_meaning[:, 100:130] = 1

#gate_lang_to_motor = np.zeros([numSeq, stepEachSeq], dtype = np.float32)
#gate_lang_to_motor[:, 30:130] = 1

#gate = np.ones([numSeq, stepEachSeq], dtype = np.float32)

print("data loaded")

############################### training iterations #########################################

MTRNN.sess.run(tf.global_variables_initializer())

flag_save = False


#plt.plot(m_train[0, :, 32], 'b')
#plt.plot(m_output[0, :, 32], 'r')
#plt.plot(m_gener[0, :, 32], 'g')
#plt.show()

#print(y_train)

#raw_input()
epoch_idx = 0
#complicated logic:
# 1) we train CS and Lang, or;
# 2) we train only Lang, or;
# 3) we train only CS.
while (alternate and (lang_loss_list[-1] > threshold_lang or motor_loss_list[-1] > threshold_motor)) or (not alternate and ((direction and lang_loss_list[-1] > threshold_lang) or (not direction and motor_loss_list[-1] > threshold_motor))): 
    print("Training epoch " + str(epoch_idx))
    if direction:
        lang_inputs = np.zeros([numSeq, stepEachSeq, lang_input], dtype = np.float32)
        motor_inputs = m_train
        motor_outputs = np.zeros([numSeq, stepEachSeq, motor_input], dtype = np.float32)#m_output
        #gate_1 = gate_lang_to_motor
        #gate_2 = gate
    else:
        lang_inputs = x_train
        motor_inputs = m_gener
        motor_outputs = m_output
        #gate_1 = gate
        #gate_2 = gate_motor_to_lang

    t0 = datetime.datetime.now()
    #_total_loss, _train_op, _state_tuple = MTRNN.sess.run([MTRNN.total_loss, MTRNN.train_op, MTRNN.state_tuple], feed_dict={MTRNN.x:lang_inputs, MTRNN.y:y_train, MTRNN.m:motor_inputs, MTRNN.m_o:motor_outputs, MTRNN.direction:direction, MTRNN.gate_motor: gate_1, MTRNN.gate_lang:gate_2, 'initU_0:0':init_state_IO_l, 'initC_0:0':init_state_IO_l, 'initU_1:0':init_state_fc_l, 'initC_1:0':init_state_fc_l, 'initU_2:0':init_state_sc_l, 'initC_2:0':init_state_sc_l, 'initU_3:0':init_state_ml, 'initC_3:0':init_state_ml, 'initU_4:0':init_state_sc_m, 'initC_4:0':init_state_sc_m, 'initU_5:0':init_state_fc_m, 'initC_5:0':init_state_fc_m, 'initU_6:0':init_state_IO_m, 'initC_6:0':init_state_IO_m})
    _total_loss, _train_op, _state_tuple = MTRNN.sess.run([MTRNN.total_loss, MTRNN.train_op, MTRNN.state_tuple], feed_dict={MTRNN.x:lang_inputs, MTRNN.y:y_train, MTRNN.m:motor_inputs, MTRNN.m_o:motor_outputs, MTRNN.direction:direction, 'initU_0:0':init_state_IO_l, 'initC_0:0':init_state_IO_l, 'initU_1:0':init_state_fc_l, 'initC_1:0':init_state_fc_l, 'initU_2:0':init_state_sc_l, 'initC_2:0':init_state_sc_l, 'initU_3:0':init_state_ml, 'initC_3:0':init_state_ml, 'initU_4:0':init_state_sc_m, 'initC_4:0':init_state_sc_m, 'initU_5:0':init_state_fc_m, 'initC_5:0':init_state_fc_m, 'initU_6:0':init_state_IO_m, 'initC_6:0':init_state_IO_m})
    t1 = datetime.datetime.now()
    print("epoch time: ", (t1-t0).total_seconds())
    if direction:
        loss = _total_loss
        print("training sentences: ", loss)
        new_loss = loss
        #if loss > 5:
        #    new_loss = 5
        lang_loss_list.append(new_loss)
    else:
        loss = _total_loss
        print("training CS: ", loss)
        new_loss = loss
        #if loss > 5:
        #    new_loss = 5
        motor_loss_list.append(new_loss)
    if epoch_idx%2 == 0:
        average_loss = alpha*lang_loss_list[-1] + (1-alpha)*motor_loss_list[-1]
    loss_list.append(average_loss)
    print("Current best loss: ",best_loss)
    print("#################################")
    print("epoch "+str(epoch_idx)+", loss: "+str(loss))
    if lang_loss_list[-1] <= best_loss_lang and motor_loss_list[-1] <= best_loss_motor:
        model_path = my_path + "/mtrnn_"+str(epoch_idx) + "_loss_" + str(average_loss)
        save_path = MTRNN.saver.save(MTRNN.sess, model_path)
        best_loss_lang = lang_loss_list[-1]
        best_loss_motor = motor_loss_list[-1]
        best_loss = alpha*lang_loss_list[-1] + (1-alpha)*motor_loss_list[-1]
        flag_save =True
    epoch_idx += 1

    if alternate:
        if motor_loss_list[-1] < 2*lang_loss_list[-1] or motor_loss_list[-1] < threshold_motor:
            direction = True
            if epoch_idx%10 == 0:
                direction = not direction

        if lang_loss_list[-1] < 2*motor_loss_list[-1] or lang_loss_list[-1] < threshold_lang:
            direction = False
            if epoch_idx%10 == 0:
                direction = not direction

    t2 = datetime.datetime.now()
    print("saving time: ", (t2-t1).total_seconds())
    if epoch_idx > NEPOCH:
        break

##################################### Print error graph ####################################
plt.ion()
fig = plt.figure()
ax = plt.subplot(1,1,1)
fig.show()
plot(loss_list, fig, ax)
####################### FOR TEST PURPOSES ONLY ######################################
if not flag_save:
    model_path = my_path + "/mtrnn_"+str(epoch_idx) + "_loss_" + str(average_loss)
    save_path = MTRNN.saver.save(MTRNN.sess, model_path)
#####################################################################################

########################################## TEST ############################################

MTRNN.saver.restore(MTRNN.sess, save_path)
plt.ioff()
plt.show()
print("testing")

PRINT_TABLE = False


init_state_IO_l = np.zeros([1, input_layer], dtype = np.float32)
init_state_fc_l = np.zeros([1, lang_dim1], dtype = np.float32)
init_state_sc_l = np.zeros([1, lang_dim2], dtype = np.float32)
init_state_ml = np.zeros([1, meaning_dim], dtype = np.float32)
init_state_IO_m = np.zeros([1, motor_layer], dtype = np.float32)
init_state_fc_m = np.zeros([1, motor_dim1], dtype = np.float32)
init_state_sc_m = np.zeros([1, motor_dim2], dtype = np.float32)

test_false = False
test_true = True

MTRNN.forward_step_test()

tf.get_default_graph().finalize()
States = np.zeros([10, stepEachSeq, 8, lang_dim1], dtype = np.float32) # 3 layers + Input + output

b=0
for i in range(0, 1, jumps):
    #raw_input()
    #new_output = np.asarray(np.zeros((1, stepEachSeq)),dtype=np.int32)
    #new_input = np.asarray(np.zeros((1, stepEachSeq, lang_dim2)),dtype=np.float32)
    #new_sentence = np.asarray(np.zeros((1, stepEachSeq, lang_input)), dtype=np.float32)
    #new_final_seq = np.asarray(np.zeros((1, control_dim)), dtype=np.float32)
    #new_output[0, :] = old_y[i, :]
    
    #new_final_seq[0,:] = old_control[i, 0, 0:8]
    new_lang_out = np.asarray(np.zeros((1, stepEachSeq)),dtype=np.int32)
    new_motor_in = np.asarray(np.zeros((1, stepEachSeq, motor_input)),dtype=np.float32)
    new_lang_in = np.asarray(np.zeros((1, stepEachSeq, lang_input)), dtype=np.float32)
    new_motor_out = np.asarray(np.zeros((1, stepEachSeq, motor_input)), dtype=np.float32)

    print("sentence: ", sentence_list[i])

    if test_true:
        direction = True
        #t0 = datetime.datetime.now()
        new_motor_in[0, :, :] = m_train[i, :, :]
        #state_list = []
        #output_list = []
        softmax_list = np.zeros([stepEachSeq, lang_input], dtype = np.float32)

        input_x = np.zeros([1, motor_input], dtype = np.float32)
        input_sentence = np.zeros([1, lang_input], dtype = np.float32)
        State = ((init_state_IO_l, init_state_IO_l), (init_state_fc_l, init_state_fc_l), (init_state_sc_l, init_state_sc_l), (init_state_ml, init_state_ml), (init_state_sc_m, init_state_sc_m), (init_state_fc_m, init_state_fc_m),(init_state_IO_m, init_state_IO_m))        
        ################################################
        
        for l in range(stepEachSeq):
            input_x[0,:] = new_motor_in[0,l,:]
            input_sentence[0,:] = new_lang_in[0,l,:]
            init_state_00 = State[0][0]
            init_state_01 = State[0][1]
            init_state_10 = State[1][0]
            init_state_11 = State[1][1]
            init_state_20 = State[2][0]
            init_state_21 = State[2][1]
            init_state_30 = State[3][0]
            init_state_31 = State[3][1]
            init_state_40 = State[4][0]
            init_state_41 = State[4][1]
            init_state_50 = State[5][0]
            init_state_51 = State[5][1]
            init_state_60 = State[6][0]
            init_state_61 = State[6][1]
            if stepEachSeq < 30:
                gate1 = np.ones([1], dtype = np.float32)
                gate2 = np.zeros([1], dtype = np.float32)
            else:
                gate1 = np.zeros([1], dtype = np.float32)
                gate2 = np.ones([1], dtype = np.float32)

            #outputs, new_state, softmax = MTRNN.sess.run([MTRNN.outputs, MTRNN.new_state, MTRNN.softmax], feed_dict = {MTRNN.direction: direction, MTRNN.Inputs_m_t: input_x, MTRNN.Inputs_sentence_t: input_sentence, MTRNN.gate_lang_t:gate1  , MTRNN.gate_motor_t:gate2, 'test/initU_0:0':init_state_01, 'test/initC_0:0':init_state_00, 'test/initU_1:0':init_state_11, 'test/initC_1:0':init_state_10, 'test/initU_2:0':init_state_21, 'test/initC_2:0':init_state_20, 'test/initU_3:0':init_state_31, 'test/initC_3:0':init_state_30, 'test/initU_4:0':init_state_41, 'test/initC_4:0':init_state_40, 'test/initU_5:0':init_state_51, 'test/initC_5:0':init_state_50, 'test/initU_6:0':init_state_61, 'test/initC_6:0':init_state_60})
            outputs, new_state, softmax = MTRNN.sess.run([MTRNN.outputs, MTRNN.new_state, MTRNN.softmax], feed_dict = {MTRNN.direction: direction, MTRNN.Inputs_m_t: input_x, MTRNN.Inputs_sentence_t: input_sentence, 'test/initU_0:0':init_state_01, 'test/initC_0:0':init_state_00, 'test/initU_1:0':init_state_11, 'test/initC_1:0':init_state_10, 'test/initU_2:0':init_state_21, 'test/initC_2:0':init_state_20, 'test/initU_3:0':init_state_31, 'test/initC_3:0':init_state_30, 'test/initU_4:0':init_state_41, 'test/initC_4:0':init_state_40, 'test/initU_5:0':init_state_51, 'test/initC_5:0':init_state_50, 'test/initU_6:0':init_state_61, 'test/initC_6:0':init_state_60})

            #t00 = datetime.datetime.now()
            softmax_list[l, :] = softmax
            State = new_state
            #t01 = datetime.datetime.now()
            #print("matrix store time: ", (t01-t00).total_seconds())
            States[b, l, 0, 0:lang_input] = States[b, l, 0, 0:lang_input] + softmax_list[l,:]
            States[b, l, 1, 0:input_layer] = States[b, l, 1, 0:input_layer] + new_state[0][1]
            States[b, l, 2, 0:lang_dim1] = States[b, l, 2, 0:lang_dim1] + new_state[1][1]
            States[b, l, 3, 0:lang_dim2] = States[b, l, 3, 0:lang_dim2] + new_state[2][1]
            States[b, l, 4, 0:meaning_dim] = States[b, l, 4, 0:meaning_dim] + new_state[3][1]
            States[b, l, 5, 0:motor_dim2] = States[b, l, 5, 0:motor_dim2] + new_state[4][1]
            States[b, l, 6, 0:motor_dim1] = States[b, l, 6, 0:motor_dim1] + new_state[5][1]
            States[b, l, 7, 0:motor_layer] = States[b, l, 7, 0:motor_layer] + new_state[6][1]

            
        sentence = ""
        #print("Sequence with new model:", new_motor_in[:,0,:])
        for t in range(stepEachSeq):
            for g in range(lang_input):
                if softmax_list[t,g] == max(softmax_list[t]): 
                    if g <26:
                        sentence += chr(97 + g)
                    if g == 26:
                        sentence += " "
                    if g == 27:
                        sentence += "."
################################# Print table #####################################
        if PRINT_TABLE:        
            color = 0

            fig, ax = plt.subplots()
            Mat = np.transpose(softmax_list[:,0:lang_input])
            print(np.shape(Mat))
            cax = ax.matshow(Mat, cmap=plt.cm.binary, vmin = 0, vmax = 1)
            cbar = fig.colorbar(cax, ticks = [0, 1])
            cbar.ax.set_yticklabels(['0', '1'])
            #plt.grid(b = True, which = 'major', color = 'black', linestyle = '-')
            for t in range(lang_input+1):
                ax.axhline(y=t+0.5, ls='-', color='black')
                if t < 26:
                    plt.text(-2,t+0.5,str(chr(97+t)))
                if t == 26:
                    plt.text(-2,t+0.5," ")
                if t == 27:
                    plt.text(-2,t+0.5,".")
            for t in range(stepEachSeq+1):
                ax.axvline(x=t+0.5, ls='-', color='black')
            plt.xlabel("timesteps");
            ax.set_yticklabels([])
            plt.show()
 
        print("output: ",sentence)
        print("#######################################")
        sentence = ""
        for g in range(stepEachSeq):
            if y_train[i,g] == 26:
                sentence += " "
            elif y_train[i,g] == 27:
                sentence += "."
            else:
                sentence += chr(97 + y_train[i,g])

        print("target: " ,sentence)
        print("#######################################")


        #if RUN_PCA:
            #execute_pca(old_sentence, States, lang_input, input_layer, lang_dim1, lang_dim2, control_dim, direction, b, i-9)


    if test_false:
        direction = False
        new_motor_in[0, :, :] = m_gener[i, :, :]
        new_lang_in[0,:,:] = x_train[i,:,:]
        #state_list = []
        output_list = []
        #softmax_list = np.zeros([stepEachSeq, lang_input], dtype = np.float32)

        input_x = np.zeros([1, motor_input], dtype = np.float32)
        input_sentence = np.zeros([1, lang_input], dtype = np.float32)
        #t1 = datetime.datetime.now()
        #print("sentence test time: ", (t1-t0).total_seconds())

        softmax_list = np.zeros([stepEachSeq, lang_input], dtype = np.float32)

        input_x = np.zeros([1, motor_input], dtype = np.float32)
        input_sentence = np.zeros([1, lang_input], dtype = np.float32)
        State = ((init_state_IO_l, init_state_IO_l), (init_state_fc_l, init_state_fc_l), (init_state_sc_l, init_state_sc_l), (init_state_ml, init_state_ml), (init_state_sc_m, init_state_sc_m), (init_state_fc_m, init_state_fc_m),(init_state_IO_m, init_state_IO_m))
        ################################################
        
        for l in range(stepEachSeq):
            input_x[0,:] = new_motor_in[0,l,:]
            input_sentence[0,:] = new_lang_in[0,l,:]
            init_state_00 = State[0][0]
            init_state_01 = State[0][1]
            init_state_10 = State[1][0]
            init_state_11 = State[1][1]
            init_state_20 = State[2][0]
            init_state_21 = State[2][1]
            init_state_30 = State[3][0]
            init_state_31 = State[3][1]
            init_state_40 = State[4][0]
            init_state_41 = State[4][1]
            init_state_50 = State[5][0]
            init_state_51 = State[5][1]
            init_state_60 = State[6][0]
            init_state_61 = State[6][1]
            if stepEachSeq < 100:
                gate1 = np.zeros([1], dtype = np.float32)
                gate2 = np.ones([1], dtype = np.float32)
            else:
                gate1 = np.ones([1], dtype = np.float32)
                gate2 = np.zeros([1], dtype = np.float32)
            #outputs, new_state = MTRNN.sess.run([MTRNN.outputs, MTRNN.new_state], feed_dict = {MTRNN.direction: direction, MTRNN.Inputs_m_t: input_x, MTRNN.Inputs_sentence_t: input_sentence, MTRNN.gate_lang_t:gate1  , MTRNN.gate_motor_t:gate2, 'test/initU_0:0':init_state_01, 'test/initC_0:0':init_state_00, 'test/initU_1:0':init_state_11, 'test/initC_1:0':init_state_10, 'test/initU_2:0':init_state_21, 'test/initC_2:0':init_state_20, 'test/initU_3:0':init_state_31, 'test/initC_3:0':init_state_30, 'test/initU_4:0':init_state_41, 'test/initC_4:0':init_state_40, 'test/initU_5:0':init_state_51, 'test/initC_5:0':init_state_50, 'test/initU_6:0':init_state_61, 'test/initC_6:0':init_state_60})
            outputs, new_state = MTRNN.sess.run([MTRNN.outputs, MTRNN.new_state], feed_dict = {MTRNN.direction: direction, MTRNN.Inputs_m_t: input_x, MTRNN.Inputs_sentence_t: input_sentence, 'test/initU_0:0':init_state_01, 'test/initC_0:0':init_state_00, 'test/initU_1:0':init_state_11, 'test/initC_1:0':init_state_10, 'test/initU_2:0':init_state_21, 'test/initC_2:0':init_state_20, 'test/initU_3:0':init_state_31, 'test/initC_3:0':init_state_30, 'test/initU_4:0':init_state_41, 'test/initC_4:0':init_state_40, 'test/initU_5:0':init_state_51, 'test/initC_5:0':init_state_50, 'test/initU_6:0':init_state_61, 'test/initC_6:0':init_state_60})
            output_list += [outputs]

            #t00 = datetime.datetime.now()
            State = new_state
            #t01 = datetime.datetime.now()
            #print("matrix store time: ", (t01-t00).total_seconds())
            States[b, l, 0, 0:lang_input] = States[b, l, 0, 0:lang_input] + softmax_list[l,:]
            States[b, l, 1, 0:input_layer] = States[b, l, 1, 0:input_layer] + new_state[0][1]
            States[b, l, 2, 0:lang_dim1] = States[b, l, 2, 0:lang_dim1] + new_state[1][1]
            States[b, l, 3, 0:lang_dim2] = States[b, l, 3, 0:lang_dim2] + new_state[2][1]
            States[b, l, 4, 0:meaning_dim] = States[b, l, 4, 0:meaning_dim] + new_state[3][1]
            States[b, l, 5, 0:motor_dim2] = States[b, l, 5, 0:motor_dim2] + new_state[4][1]
            States[b, l, 6, 0:motor_dim1] = States[b, l, 6, 0:motor_dim1] + new_state[5][1]
            States[b, l, 7, 0:motor_layer] = States[b, l, 7, 0:motor_layer] + new_state[6][1]

        output_vec = np.zeros([stepEachSeq, motor_input], dtype = np.float32)
        #print(np.shape(output_list[0][0][0]))
        #print(np.shape(output_vec))
        for t in range(len(output_list)):
            output_vec[t,:] = output_list[t][0][0][0:motor_input]
        #print(output_vec)
        for t in range(1, motor_input, 3):
            plt.plot(output_vec[:,t], 'r')
            plt.plot(m_output[i, :, t], 'b')
            plt.show()

        #if RUN_PCA:
            #execute_pca(old_sentence, States, lang_input, input_layer, lang_dim1, lang_dim2, control_dim, direction, b, i-9)
        #t2 = datetime.datetime.now()
        #print("cs test time: ", (t2-t1).total_seconds())
        print("\n")
        print("\n")
    b+= 1
    if (i+1)%10 == 0 and i != 0:
        #execute_pca(old_sentence, States, lang_input, input_layer, lang_dim1, lang_dim2, control_dim, direction, b, i-9)
        b = 0
        #execute_pca(old_sentence, States, lang_input, input_layer, lang_dim1, lang_dim2, control_dim, direction, b+1)


MTRNN.sess.close()
#MTRNNTest.sess.close()

