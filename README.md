# Part-of-Speech-Tagging

<h2>Naive Bayes Method</h2>

![Bayes model](https://github.com/santoshd97/Part-of-Speech-Tagging/blob/master/images/Bayes.png)

<ul>
<li>We use the following Bayes formula to find probability:
P(Si|Wi) = P(Wi|Si) * P(Si)/P(Wi)
</li>
<li>Since the denominator will remain same for given word, we ignore the P(Wi).
P(Si|Wi) = P(Wi|Si) * P(Si)
</li>
<li>For each word Wi in test sentence, the part of speech tag Si with the maximum probability P(Si|Wi) is considered for classifying the word Wi </li>

<li> Handling new words in simple model: If a word in the test sentence is not present in emission probability calculated from training data then we assign emission probability to that word = (minimum of all emission probabilities)*0.1 </li>
</ul>

<h2>Hidden Markov Model(HMM) using Viterbi</h2>

![HMM model](https://github.com/santoshd97/Part-of-Speech-Tagging/blob/master/images/HMM.png)

<ul>
<li>Viterbi is based on dynamic programming</li>
<li>The number of states in Viterbi model = number of words in test sentence</li>
<li>For the first word, we only calculate emission probability * prior probability for all 12 parts of speech and save the values in dictionary v["state, part of speech"] </li>
<li>From second word onwards, we calculate max for each of 12 part of speech = max(previous Viterbi state * transition probability). Corresponding part of speech and pervious part of speech is stored in a list of list called tag. Then we multiply this value with corresponding emission probability and get value for 2nd Viterbi state</li>
<li>This process continues till last word of sentence </li>
<li>We find max value of last Viterbi state which is stored in dictionary v and get corresponding part of speech tag which is the tag for last word in sentence. We then backtrack using this part of speech to find tags for each word </li> 

<li>Handling emission probabilities of new words by assigning min(emission probability)*0.1</li>

<li>We have first taken log of individual probabilities (emission, transition) and then added them instead of multiplication given in formula to handle underflow problem </li>
</ul>
  
<h2>Markov Chain Monte Carlo(MCMC)</h2>

![MCMC model](https://github.com/santoshd97/Part-of-Speech-Tagging/blob/master/images/MCMC.png)

<ul>
<li>First, generate a random sample consisting of all nouns. From this sample, we generate a new sample using create_sample() function </li>
<li>To generate a sample, the probability of a part of speech tag for a word is calculated, given all the other words and their corresponding tags observed, that is P(S_i | (S - {S_i}), W1,W2,...,Wn) </li>
<li>For first word we calculated product of emission and prior probability </li>
<li>For the second word we calculated product of emission and transition of tag from 1st to 2nd word </li>
<li>From 3rd word onwards, we calculated product of emission, transition from previous to current tag, and transition from previous of previous to current tag </li>
<li>We then divide these probabilities for each tag of given word by sum of all probabilities of all tags for that word. Using np.random.choice we randomly get a tag having probability close to 1 and assign this tag to that word in sample. We do this process for all samples </li>
<li>To handle emission probability of new words, a small probability (min(emission)*0.1) is assigned. The first 1000 samples were discarded to pass the warming period and improve the accuracy </li>
<li>To calculate marginal distribution from samples, we first generated 3000 samples using MCMC (after discarding the first 1000 samples) </li>
<li>From these samples we calculate max probability of each part of speech tag corresponding to each word in the test sentence </li>
<li>We assign the part of speech tag having maximum probability for a word and combine them to get the part of speech tags for the entire sentence </li>
<li>Handling emission probabilities of new words by assigning min(emission probability)*0.1</li>
</ul>

<h2>Result</h2>
Command to run code ./label.py bc.train.txt bc.test.txt
<h3>Naive Bayes Accuracy</h3>
Words correct = 93.92% <br>
Sentences correct = 47.45%

<h3>HMM using Viterbi Accuracy</h3>
Words correct = 94.70% <br>
Sentences correct = 53.25%

<h3>MCMC Accuracy</h3>
Words correct = 95.08% <br>
Sentences correct = 55.65%
