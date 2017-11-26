import numpy as np
from copy import deepcopy
import re
import pickle
from functools import reduce
from scipy import random
import operator

class hmm:
    def checkvalue(self,value):
        #print value
        pass

    def __init__(self):
        self.transition_matrix = np.empty([0])
        self.initial_matrix = np.empty([0])
        self.emission_matrix = {}
        self.number_of_states = 0
        self.state_tag_map = np.array([])

    def randomize(self,observations,num_states):
        
        self.number_of_states = num_states
        self.transition_matrix = random.dirichlet(np.ones((num_states)),size=num_states)
        self.checkvalue(self.transition_matrix)
        self.transition_matrix.dtype = np.float64
        self.initial_matrix = random.dirichlet(np.ones((num_states)))
        self.checkvalue(self.initial_matrix)
        self.initial_matrix.dtype = np.float64
        
        for o in observations: 
            self.emission_matrix[o] = random.dirichlet(np.ones((num_states)))
            self.checkvalue(self.emission_matrix)
            self.emission_matrix[o].dtype = np.float64
    
    def _forward_algorithm(self,arr):
        forward_array = np.zeros([len(arr),self.number_of_states],dtype=np.float64)
        self.checkvalue(forward_array)
        forward_array[0]=self.emission_matrix[arr[0]]*self.initial_matrix
        self.checkvalue(forward_array[0])
        for index,entry in enumerate(arr[1:]):                     
            forward_array[index+1] = (forward_array[index] * self.emission_matrix[entry].reshape((self.number_of_states,1)) * self.transition_matrix.T).sum(1)
            self.checkvalue(forward_array[index+1])
        return forward_array
                                     
    def _backward_algorithm(self,arr):
        arr = arr[::-1]
        backward_array = np.full([len(arr), self.number_of_states],1,dtype=np.float64)
        self.checkvalue(backward_array)
        for index,entry in enumerate(arr[:-1]):
            backward_array[index+1] = (backward_array[index] * self.emission_matrix[entry] * self.transition_matrix).sum(1)
            self.checkvalue(backward_array[index+1])
        return backward_array[::-1]

    def baum_welch(self,arrs):
        current_probs = np.full((len(arrs)),0.0000000001,dtype=np.float64)
        self.checkvalue(current_probs)
        last_probs = np.zeros([len(arrs)],dtype=np.float64)
        self.checkvalue(last_probs)
        while current_probs.sum() > last_probs.sum():
            last_probs = current_probs.copy()
            self.checkvalue(last_probs)
            initial_m = np.zeros(self.initial_matrix.shape,dtype=np.float64)
            self.checkvalue(initial_m)
            transition_m = np.zeros(self.transition_matrix.shape,dtype=np.float64)
            emission_m = deepcopy(self.emission_matrix)
            self.checkvalue(emission_m)
            for i,arr in enumerate(arrs):
                if i%int(len(arrs)*0.1) == 0:
                    print(i/float(len(arrs)))
                f_array = self._forward_algorithm(arr)
                self.checkvalue(f_array)
                b_array = self._backward_algorithm(arr)
                self.checkvalue(b_array)
                probability_of_arr = f_array[-1].sum()
                
                
                current_probs[i] = probability_of_arr                
                
                if probability_of_arr != 0.0:
                    prob_of_transitions = np.zeros([len(arr),self.number_of_states,self.number_of_states])
                    self.checkvalue(prob_of_transitions)
                    prob_of_states = np.empty([len(arr),self.number_of_states])
                    
                    for t in range(len(arr)-1):
                        prob_of_transitions[t] = f_array[t].reshape(self.number_of_states,1)*self.transition_matrix * b_array[t+1]*self.emission_matrix[arr[t+1]]  
                        self.checkvalue(prob_of_transitions[t])
                    prob_of_transitions = prob_of_transitions/probability_of_arr
    
                    prob_of_states = prob_of_transitions.sum(2)
                    self.checkvalue(prob_of_states)
                    prob_of_states[-1] = f_array[-1]/probability_of_arr
                    
                    prob_of_transitions = prob_of_transitions / prob_of_states.reshape((len(arr),self.number_of_states,1))
                    self.checkvalue(prob_of_transitions)
                    initial_m += prob_of_states[0]
                    
                    transition_m += prob_of_transitions.sum(0)
                    self.checkvalue(transition_m)
                    for entry in self.emission_matrix:
                        boolean_array = np.equal(np.array(arr,dtype=np.object),entry)
                        self.checkvalue(boolean_array)
                        emission_m[entry] += (prob_of_states*boolean_array.reshape(len(arr),1)/(prob_of_states.sum(0))).sum(0)
                        
                        
                        
            print('current probs:')
            print(current_probs.sum()) 
            print(current_probs.mean())
            print('last probs:')
            print(last_probs.sum()) 
            print(last_probs.mean())            
            initial_m = initial_m/initial_m.sum()
            transition_m = transition_m/transition_m.sum(1)
            sums = np.zeros((self.number_of_states),dtype=np.float64)
            self.checkvalue(sums)
            for e,a in emission_m.items():
                sums += a
        
            for e in emission_m:
                emission_m[e] = emission_m[e]/sums
                self.checkvalue(emission_m[e])
            self.initial_matrix = initial_m
            self.transition_matrix = transition_m
            self.emission_matrix = emission_m
            tag = [[] for i in range(10)]
            for e in emission_m:
                for i in range(len(emission_m[e])):

                    tag[i].append((emission_m[e][i],e))
            for ta in tag:
                sortta = sorted(ta, key=operator.itemgetter(0),reverse=True)
                for i in sortta[0:100]:
                    print i[1],
                print "next"
                    
h = hmm()
h.randomize(set(brown.words()),10)
h.baum_welch(brown.sents()[:1000])
