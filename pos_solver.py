###################################
# CS B551 Fall 2018, Assignment #3
#
# Santoshmurti Daptardar (sdaptard)
#
#######
""""
REPORT 

# Training data:

  We have created the following dictionaries for calculating probabilities:
	1. self.ps - Prior probability P(S1) i.e. probability of parts of speech tags = (frequency of tag S) / (total words in training set)
	2. self.transition - Transition probability P(S i+1|S i) i.e. probability of transition from one part of speech to another part of speech = (frequency that Si is followed by Si+1) / (total occurrences of Si in training set)
	3. self.wsprob - Emission probability P(W i|S i) i.e. probability of a word given a specific part of speech = (frequency of word Wi being part of speech Si) / (total occurrences of part of speech Si in training set)


  # Missing values in training data:

	If a particular transition from one part of speech to another has missing value in training data then we assume the probability to be min(transition probability)*0.1

	If a particular emission has missing value then we assume the probability to be min(emission probability)*0.1
	
# Simple model

We use the following Bayes formula to find probability:
P(Si|Wi) = P(Wi|Si) * P(Si)/P(Wi)

Since the denominator will remain same for given word, we ignore the P(Wi).
P(Si|Wi) = P(Wi|Si) * P(Si)

For each word Wi in test sentence, the part of speech tag Si with the maximum probability P(Si|Wi) is considered for classifying the word Wi

    # Handling new words in simple model

	If a word in the test sentence is not present in emission probability calculated from training data then we assign emission probability to that word = (minimum of all emission probabilities)*0.1


# Viterbi model

Viterbi is based on dynamic programming.
The number of states in Viterbi model = number of words in test sentence
For the first word, we only calculate emission probability * prior probability for all 12 parts of speech and save the values in dictionary v["state, part of speech"]
From second word onwards, we calculate max for each of 12 part of speech = max(previous Viterbi state * transition probability). Corresponding part of speech and pervious part of speech is stored in a list of list called tag. Then we multiply this value with corresponding emission probability and get value for 2nd Viterbi state.
This process continues till last word of sentence. 
We find max value of last Viterbi state which is stored in dictionary v and get corresponding part of speech tag which is the tag for last word in sentence. We then backtrack using this part of speech to find tags for each word.  

# Handling emission probabilities of new words by assigning min(emission probability)*0.1

# We have first taken log of individual probabilities (emission, transition) and then added them instead of multiplication given in formula to handle underflow problem. 

# Complex model (MCMC)

First, generate a random sample consisting of all nouns. From this sample, we generate a new sample using create_sample() function. To generate a sample, the probability of a part of speech tag for a word is calculated, given all the other words and their corresponding tags observed, that is P(S_i | (S - {S_i}), W1,W2,...,Wn). For first word we calculated product of emission and prior probability. For the second word we calculated product of emission and transition of tag from 1st to 2nd word. From 3rd word onwards, we calculated product of emission, transition from previous to current tag, and transition from previous of previous to current tag. We then divide these probabilities for each tag of given word by sum of all probabilities of all tags for that word. Using np.random.choice we randomly get a tag having probability close to 1 and assign this tag to that word in sample. We do this process for all samples. To handle emission probability of new words, a small probability (min(emission)*0.1) is assigned. The first 1000 samples were discarded to pass the warming period and improve the accuracy.
To calculate marginal distribution from samples, we first generated 3000 samples using MCMC (after discarding the first 1000 samples). From these samples we calculate max probability of each part of speech tag corresponding to each word in the test sentence. We assign the part of speech tag having maximum probability for a word and combine them to get the part of speech tags for the entire sentence.

Handling emission probabilities of new words by assigning min(emission probability)*0.1
 

# Accuracy

 1. Simple model: Words correct = 93.92% 
		  Sentences correct = 47.45%

 2. Viterbi model: Words correct = 94.70%
		   Sentences correct = 53.25%

 3. Complex model: Words correct = 93.15%
		   Sentences correct = 45.15% 


# Problems faced

It is very difficult to correctly classify a word that has never occurred in our training set. Assigning emission probabilities to new words (by giving any small random value) is challenging and different values  result in change in accuracy. Training set contains approx 45,000 sentences. More training data could have resulted in better accuracy.

"""
#######

import random
import math
import numpy as np

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#

class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
		p_value = 1.0
		for s,l in zip(sentence, label):
        		p_value = float(p_value) * self.post_p[l+"|"+s]
		return (math.log(p_value))
        elif model == "Complex":
		po = 0 
        	for s in xrange(len(sentence)):
			for ww1 in self.w1:
				if sentence[s] == ww1:
					em = self.wsprob[sentence[s]+"|"+label[s]]
				else:
					em = self.min_word_prob * 0.1
			if s == 0:
				po = em * self.ps[label[s]]
			elif s == 1:
				po *= em * self.transition[label[s]+"|"+label[s-1]]
			else:
				po *= em * self.transition[label[s]+"|"+label[s-1]] * self.transition3[label[s]+"|"+label[s-2]]

		return math.log(po)
        elif model == "HMM":		
        	return self.viter[str(len(sentence))+","+label[len(label)-1]]
        else:
        	print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):
    		
	pos = ["det","adj","adv","adp","conj","noun","num","pron","prt","verb","x","."]
	
	# P(S1)
	ps1 = dict()
	count = dict()
	total_words = 0
	for r in range(0,len(data)):
		total_words += len(data[r][0])
	for j1 in range(0,len(pos)):
		ps1[pos[j1]] = 0
		count[pos[j1]] = 0
	for j2 in range(0,len(data)):
		for p in range(0,len(data[j2][1])):
			ps1[data[j2][1][p]] = ps1[data[j2][1][p]] + 1
			count[data[j2][1][p]] = count[data[j2][1][p]] + 1
	for p1 in ps1:
		ps1[p1] = float(ps1[p1])/total_words
	#for x in ps1:
		#print (x, ps1[x])
	self.ps = ps1

	# Transitional probability P(S i+1|S i)
	trans_prob = dict()
	for j in range(0,len(pos)):
		for p in range(j,len(pos)):
			trans_prob[pos[j]+"|"+pos[p]] = 0
	for n1 in range(len(pos)-1,-1,-1):
		for n2 in range(n1-1,-1,-1):
			trans_prob[pos[n1]+"|"+pos[n2]] = 0 
	for k in range(0,len(data)):
		for m in range(1,len(data[k][1])):
			trans_prob[data[k][1][m]+"|"+data[k][1][m-1]] += 1
	min_trans = dict()
	for t in trans_prob:
		t1 = t.split("|")
		trans_prob[t] = float(trans_prob[t])/count[t1[1]]
		if trans_prob[t] != 0:
			min_trans[t] = trans_prob[t]
	min_min_trans = min(min_trans.values())
	for t in trans_prob:
		if trans_prob[t] == 0:
			trans_prob[t] = min_min_trans * 0.1
	#for t in trans_prob:
		#print(t,trans_prob[t])
	#print(len(trans_prob))

	self.transition = trans_prob

	# Transition probability for state 1 to state 3
	trans3 = dict()
	for j in range(0,len(pos)):
		for p in range(j,len(pos)):
			trans3[pos[j]+"|"+pos[p]] = 0
	for n1 in range(len(pos)-1,-1,-1):
		for n2 in range(n1-1,-1,-1):
			trans3[pos[n1]+"|"+pos[n2]] = 0 
	for k in range(0,len(data)):
		for m in range(2,len(data[k][1])):
			trans3[data[k][1][m]+"|"+data[k][1][m-2]] += 1
	min_trans3 = dict()
	for t in trans3:
		t3 = t.split("|")
		trans3[t] = float(trans3[t])/count[t3[1]]
		if trans3[t] != 0:
			min_trans3[t] = trans3[t]
	min_min_trans3 = min(min_trans3.values())
	for t in trans3:
		if trans3[t] == 0:
			trans3[t] = min_min_trans3 * 0.1
	self.transition3 = trans3	

	# Probability of word being a particular part of speech P(W i| S i)
	ws_prob = dict()
	for k in range(0,len(data)):
		for i in range(0,len(data[k][0])):
			for j in range(0,len(pos)):
				ws_prob[data[k][0][i]+"|"+pos[j]] = 0
	for k in range(0,len(data)):
		for i in range(0,len(data[k][0])):
			ws_prob[data[k][0][i]+"|"+data[k][1][i]] += 1
	min_em = dict()
	for w in ws_prob:
		w1 = w.split("|")
		ws_prob[w] = float(ws_prob[w])/count[w1[1]]
		if ws_prob[w] != 0:
			min_em[w] = ws_prob[w]
	mini = min(min_em.values()) 
	#print mini
	for w in ws_prob:
		if ws_prob[w]==0:
			ws_prob[w] = mini * 0.1
	#print(ws_prob["!|noun"])

	self.wsprob = ws_prob
	self.min_word_prob = min(self.wsprob.values())

	w2 = []
	for ws in self.wsprob.keys():
		ws1 = ws.split("|")	
		w2.extend([ws1[0]])
	self.w1 = w2

	
    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        
	poster = dict()
	poster_copy = dict()
	result = list()
	pos = ["det","adj","adv","adp","conj","noun","num","pron","prt","verb","x","."]
	for s in sentence:
		cc = 0
		for ww1 in self.w1:
			if s == ww1: 
				cc += 1
				break
				
		if cc==0:
			for p in xrange(0,len(pos)):
				poster[pos[p]+"|"+s] = self.min_word_prob * 0.1 * self.ps[pos[p]]
				poster_copy[pos[p]+"|"+s] = poster[pos[p]+"|"+s]
			max_value = max(poster.values())
			for key, v in poster.items():
				if v == max_value:
					k = key.split("|")
					result.extend([k[0]])
					break
		else:
			for p in xrange(0,len(pos)):
				poster[pos[p]+"|"+s] = self.wsprob[s+"|"+pos[p]] * self.ps[pos[p]]	
				poster_copy[pos[p]+"|"+s] = poster[pos[p]+"|"+s]
			max_value = max(poster.values())
			for key, v in poster.items():
				if v == max_value:
					k = key.split("|")
					result.extend([k[0]])
					break
		poster.clear()
	self.post_p = poster_copy
	return result

	
    def create_sample(self, sentence, sample, word_found): 

	pos = ["det","adj","adv","adp","conj","noun","num","pron","prt","verb","x","."]
	sentence_len = len(sentence)
	for index in xrange(sentence_len):
        	word = sentence[index]
        	probabilities = [0.0] * len(pos)
		
		if index > 0:
        		s_1 = str(sample[index - 1])
	  	if index > 1:
			s_2 = str(sample[index - 2])

	    	for j in xrange(len(pos)): 
                	s_3 = pos[j]
			
			if word_found[index] == 1:
				emission = self.wsprob[word+"|"+s_3]
			else:
				emission = self.min_word_prob * 0.1
			
			if index == 0:
				probabilities[j] = emission * self.ps[s_3]
			elif index == 1:
				probabilities[j] = emission * self.transition[s_3+"|"+s_1]
			else:
				probabilities[j] = emission * self.transition[s_3+"|"+s_1] * self.transition3[s_3+"|"+s_2]
	
		s = sum(probabilities)
		for p in xrange(len(probabilities)):
			probabilities[p] = float(probabilities[p]) / s
		final_tag = np.random.choice(pos,1,p=probabilities)
		sample[index] = final_tag[0]

	return sample
		

    def mcmc(self, sentence, sample_count):
	
	sent_len = len(sentence)
	# Initial sample - All are 'Nouns'
	sample1 = ["noun"] * sent_len 
	word_found = [0] * sent_len
	for s in xrange(sent_len):
		for ww1 in self.w1:
        		if sentence[s] == ww1: 
                		word_found[s] = 1
	# ignore first 100 samples
        for i in xrange(100): 
        	sample1 = self.create_sample(sentence, sample1, word_found)

        next_sample1 = []
        for p in xrange(sample_count):
        	sample1 = self.create_sample(sentence, sample1, word_found)
        	next_sample1.append(tuple(sample1))

        return next_sample1

    def complex_mcmc(self, sentence):

	result = []
	sample_count = 100
        next_sample = self.mcmc(sentence, sample_count)
	pos = ["det","adj","adv","adp","conj","noun","num","pron","prt","verb","x","."]
	tag_count = dict()
	for p in pos:
		tag_count[p] = 0
	for n in xrange(len(sentence)):
		for p in pos: 
			for m in xrange(len(next_sample)):
				if next_sample[m][n] == p:
					tag_count[p] += 1
		ans = max(tag_count,key=tag_count.get)
		result.append(ans)
		for p in pos:
			tag_count[p] = 0
	return result

       # return [ "noun" ] * len(sentence)


    def hmm_viterbi(self, sentence):
	
	result = list()
	v = dict()
	state_tag = list()
	tag = [[]] * (len(sentence)-1)
	max_v = dict()
	min_value = list()
	t = 0
	max_val = 0
	pos = ["det","adj","adv","adp","conj","noun","num","pron","prt","verb","x","."]

	#min_word_prob = min(self.wsprob.values())

	for s in sentence:
		cc = 0
		for ww1 in self.w1:
			if s == ww1: 
				cc += 1
				break	
		t += 1
		for p in pos:
			if t == 1 and cc == 0:
				v[str(t)+","+p] = math.log(self.min_word_prob*0.1) + math.log(self.ps[p])
			elif t == 1 and cc != 0:
				v[str(t)+","+p] = math.log(self.wsprob[s+"|"+p]) + math.log(self.ps[p])
			else:
				for  p1 in pos:
				 	max_v[p1] = v[str(t-1)+","+p1] + math.log(self.transition[p+"|"+p1])
				max_val = max(max_v.values())
				k1 = max(max_v,key=max_v.get)
				#for key, val in max_v.items():
				#	if val == max_val:
				state_tag.extend([p+":"+k1])
				if cc == 0:
					v[str(t)+","+p] = math.log(self.min_word_prob*0.1) + math.log(self.ps[p]) + max_val
				else:
					v[str(t)+","+p] = math.log(self.wsprob[s+"|"+p]) + math.log(self.ps[p]) + max_val
				max_v.clear()
		if t > 1 and len(sentence) > 1:
			tag[t-2] = state_tag
			state_tag = []
		
	self.viter = v		

	for p in pos:
		min_value.extend([v[str(len(sentence))+","+p]])

	for p in pos:
		if v[str(len(sentence))+","+p] == max(min_value):
			ans = p
	result.extend([ans])

	if len(sentence) > 1: 
		x1 = list()
		for m in xrange(len(sentence)-2,-1,-1):
			for x in tag[m]:
				x1 = x.split(":")
				if x1[0] == ans:
					result.extend([x1[1]])
					ans = x1[1]
					break
		tag = []
		return result[::-1]	
	else:
		return result

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")
