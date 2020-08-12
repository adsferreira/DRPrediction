# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 11:16:56 2014

@author: adriano
"""

import os
import random
import csv
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numpy import sort


class DataSet:
    def __init__(self, path):
        self.path = path
        self.inputs = []
        self.targets = []
        self.inputsTesting = []
        self.targetsTesting = []
        self.all_patterns = []
        self.is_all_patterns_merged = False
     
    def merge_all_patterns(self):
        self.all_patterns = np.concatenate((self.inputs, self.inputsTesting), axis = 0)
                   
    def read_data_files(self):
        file_names = os.listdir(self.path)    
        patterns = [[None] for _ in range(len(file_names))]
        targets = [[None] for _ in range(len(file_names))]
        content_file = None
        j = 0
        
        for file_name in file_names:
            fo = open(self.path + file_name, 'r')  
            content_file = fo.readlines()
            current_pattern = self.get_pattern_from_file(content_file)
            patterns[j] = current_pattern
            current_target = self.get_target_from_file_name(file_name)
            targets[j] = current_target  
            j = j + 1
            
        self.inputs = np.array(patterns)
        self.targets = np.array(targets)        
        
    def get_pattern_from_file(self, content_file):
        attributes_in_file = content_file[3:219]   
        #print "attributes len: ", len(attributes_in_file)
        current_pattern = []        
        
        for current_line in attributes_in_file:
            #print current_line
            ref_id = current_line.index('.')
            current_attribute = current_line[ref_id - 1:ref_id + 3]
            current_pattern.append(float(current_attribute))
                    
        return current_pattern   
        
    def get_target_from_file_name(self, file_name):
        target = [None]
        target[0] = float(file_name[9:18]) 
        return target
      
    def create_patterns_set_for_testing(self, iid, fid):
        """
        Supports only the generation of testing set extracted from the end of the whole matrix
        """
        self.inputs = self.all_patterns[:iid,:]
        self.inputsTesting = self.all_patterns[iid:fid,:]
#         print(self.all_patterns.shape)
#         print(self.inputs.shape)
#         print(self.inputsTesting.shape)
        
    def create_patterns_set_for_testing2(self, iid):
        """
        Supports only the generation of testing set extracted from the end of the whole matrix
        """
        inputs = self.inputs[:iid,:]
        inputsTesting = self.inputs[iid:,:]
        targets = self.targets[:iid,:]
        targetsTesting = self.targets[iid:,:]
        self.inputs = inputs
        self.inputsTesting = inputsTesting
        self.targets = targets
        self.targetsTesting = targetsTesting
        
        #print(self.inputs.shape)
        #print(self.inputsTesting.shape)
        #print(self.targets.shape)
        #print(self.targetsTesting.shape)
                        
    def create_random_testing_set(self, nr_of_samples):
        # obtain random ids
        random_ids = random.sample(xrange(0, len(self.targets) - 1), nr_of_samples)
        random_ids.sort()
        # copy the samples with ids randomly chosen
        self.inputsTesting = self.inputs[random_ids]
        self.targetsTesting = self.targets[random_ids]
        # delete test samples from training inputs
        self.inputs = np.delete(self.inputs, random_ids, axis = 0)
        self.targets = np.delete(self.targets, random_ids, axis = 0)       
        
    def read_mat_file(self, fileName):
        return sio.loadmat(self.path + fileName)       
    
    def read_csv_file(self, fileName):
        with open(self.path + fileName, 'r') as f:
            reader = csv.reader(f)
            patterns_list = list(reader)
            self.all_patterns = np.array(patterns_list).astype(np.float)
#             patterns = np.vstack({tuple(row) for row in aux_p})
#             nr_attr = patterns.shape[1]
#             oh_pbg_ids = np.where(patterns[:, (nr_attr - 1)] == 0)
#             # delete zero pbg unit cells
#             patterns = np.delete(patterns, oh_pbg_ids, axis = 0)
#             self.inputs = np.array(patterns[:, : -1])
#             self.targets = np.array(patterns[:, -1])

#         self.split_inputs_and_targets(3)
            
    def split_inputs_and_targets(self, delimiter_id):
        self.inputs = np.array(self.all_patterns[:, :-delimiter_id])
        self.targets = np.array(self.all_patterns[:, -delimiter_id:])    
                
    def set_samples_from_mat_file(self, fileName):
        mat_ds = self.read_mat_file(fileName)
        #print mat_ds
        self.inputs = np.array(mat_ds['norm_inputs'])
        self.targets = np.array(mat_ds['norm_targets'])
        #self.inputsTesting = np.array(mat_ds['patternTest'])
        #self.targetsTesting = np.array(mat_ds['targetTest'])
        #print "inputs, targets, inputs test and target test lenght: ", (self.inputs.shape, self.targets.shape, self.inputsTesting.shape, self.targetsTesting.shape)
      
    def shuffle_dataset(self):
        dataset = np.concatenate((self.inputs, self.targets), axis = 1)
        shuffled_ds = random.shuffle(dataset)
        
        self.inputs = shuffled_ds[:, : -1].tolist()
        self.targets = shuffled_ds[:, -1].tolist()
        return None
    
    def plot_distribution(self):
        x = [int(0) for _ in range(10)]
        
        # temp: soh pra pegar acoplamentos verdadeiros do acoplador 2, nao aqueles do Carlinhos
        file_name = 'power_coupling.mat' 
        pc_eff = sio.loadmat(file_name)
        pc_eff2 = np.array(pc_eff['power_coupling2'])
        pc_eff2 = np.reshape(pc_eff2, pc_eff2.shape[0])
        self.targets = pc_eff2
        
        for i in range(len(self.targets)):
            if self.targets[i] >= 0 and self.targets[i] < 10:
                x[0] += 1
            elif self.targets[i] >= 10 and self.targets[i] < 20:
                x[1] += 1
            elif self.targets[i] >= 20 and self.targets[i] < 30:
                x[2] += 1
            elif self.targets[i] >= 30 and self.targets[i] < 40:
                x[3] += 1
            elif self.targets[i] >= 40 and self.targets[i] < 50:
                x[4] += 1
            elif self.targets[i] >= 50 and self.targets[i] < 60:
                x[5] += 1
            elif self.targets[i] >= 60 and self.targets[i] < 70:
                x[6] += 1
            elif self.targets[i] >= 70 and self.targets[i] < 80:
                x[7] += 1
            elif self.targets[i] >= 80 and self.targets[i] < 90:
                x[8] += 1
            else:
                x[9] += 1
             
#        print x   
        import matplotlib as mpl
        mpl.rc('font',family='Arial')
        
        # The slices will be ordered and plotted counter-clockwise.
        labels = ['20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%']
        #labels = ['20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%']
        labels = list(reversed(labels))
        sizes = [198, 262, 358, 512, 747, 1120, 1113]
        #sizes = [110, 130, 264, 539, 558, 212]
        sizes = list(reversed(sizes))
        
        colors = ['yellowgreen', 'gold', 'salmon', 'papayawhip', 'lightblue', 'darkkhaki', 'indianred']
        #colors = ['yellowgreen', 'gold', 'salmon', 'papayawhip', 'lightblue', 'darkkhaki']
        colors = list(reversed(colors))
        explode = (0, 0.05, 0, 0, 0, 0, 0)
        #explode = (0, 0.05, 0, 0, 0, 0)
        
        fig1, ax1 = plt.subplots()
        patches, texts, autotexts = plt.pie(sizes, explode=explode, colors=colors, pctdistance=0.80,
                                            autopct='%1.1f%%', shadow=False, startangle=180)
        # Set aspect ratio to be equal so that pie is drawn as a circle.
        for t in texts:
            t.set_size(18)
        for t in autotexts:
            t.set_size(20)
            t.set_weight('semibold')
         
        ms = 20    
        circ0 = Line2D([0], [0], linestyle="none", marker="o", markersize=ms, markerfacecolor=colors[0])
        circ1 = Line2D([0], [0], linestyle="none", marker="o", markersize=ms, markerfacecolor=colors[1])
        circ2 = Line2D([0], [0], linestyle="none", marker="o", markersize=ms, markerfacecolor=colors[2])
        circ3 = Line2D([0], [0], linestyle="none", marker="o", markersize=ms, markerfacecolor=colors[3])
        circ4 = Line2D([0], [0], linestyle="none", marker="o", markersize=ms, markerfacecolor=colors[4])
        circ5 = Line2D([0], [0], linestyle="none", marker="o", markersize=ms, markerfacecolor=colors[5])
        circ6 = Line2D([0], [0], linestyle="none", marker="o", markersize=ms, markerfacecolor=colors[6])
        #circ7 = Line2D([0], [0], linestyle="none", marker="o", markersize=ms, markerfacecolor=colors[7])
        rounded_patches = (circ0, circ1, circ2, circ3, circ4, circ5, circ6)
        plt.legend(rounded_patches, labels, bbox_to_anchor=(0.875, 1.11), loc=2, borderaxespad=1,
                   numpoints=1, prop={'size':18}, handletextpad=0.3, frameon=False, labelspacing=0.8)
        
        plt.axis('equal')
        ax1.set_position([-0.1, 0., 0.8, 0.8])
        plt.savefig('/home/adriano/EclipseProjects/ANNCoupling/src/dist_db_b9.pdf', bbox_inches='tight', dpi = 300)
        
        plt.show()        
  
# ds = DataSet('/home/adriano/Projects/ANNDispersionRelation/ann_training/')   
# ds.read_csv_file('dr_dataset.csv')  
# ds.set_inputs_and_targets(3)   
#dataSet.set_samples_from_mat_file('ds_pbgs.mat')
# print("number of training inputs: ", len(ds.inputs))
# print("number of training targets: ", len(ds.targets))
# print("\n")
#ds.create_random_testing_set(6)    
# print("number of training inputs: ", len(ds.inputs))
# print("number of training targets: ", len(ds.targets))            
# print("number of testing inputs: ", len(ds.inputsTesting))
# print("number of testing targets: ", len(ds.targetsTesting))
#dataSet.create_random_testing_set(300)
#print "\n----------\n"
#print "number of training inputs: ", len(dataSet.inputs)
#print "number of training targets: ", len(dataSet.targets)
#print "\n"
#print "number of testing inputs: ", len(dataSet.inputsTesting)
#print "number of testing targets: ", len(dataSet.targetsTesting)
#plt.plot(sort(ds.targetsTesting))
#plt.show()
