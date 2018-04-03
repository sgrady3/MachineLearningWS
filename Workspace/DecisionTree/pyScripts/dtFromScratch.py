import pandas as pd
import random
import numpy as np 
import math



## Generates random data. Modify the weights matrix to determine which attribute has the most information
## The higher the value in weights, the more information that variable would have in the decision tree
def generateRandomData(nrows = 1000):
	with open("grad_school.csv",'w') as out:
		attributes = ["GPA(>3.0)","GRE(total > 300)","Good Recommendations(>2)","Masters?","Accepted"]
		out.write(",".join(attributes))
		out.write("\n")
		for i in range(nrows):
			row = np.random.randint(0,2,4)
			str_row = map(str,row)
			weights = [100,200,300,100]
			total = np.dot(weights,row)
			accepted = 0
			if total > 300:
				accepted = 1
			if random.random() < 0.33:
				accepted = abs(accepted - 1)
			out.write(",".join(str_row))
			out.write(","+str(accepted)+"\n")
		return None

## Functions for implementing a decision tree from scratch. We won't be doing this, but perhaps these can be used to assist 
## in the theory portion. Assumes data has been preprocessed to convert all points into a yes (1) or no (0) value, but we can generalize this later.

def entropy(q):
	if (1-q) == 0 or q == 0:
		return 0.0
	else:
		return -(q*math.log(q,2)+(1-q)*math.log(1-q,2))

def remainder(data,attribute,goal):
	p = len(data[data[goal] == 1])
	n = len(data[data[goal] == 0])
	r = 0.0
	for x in [0,1]:
		p_a = len(data[(data[attribute] == x) & (data[goal] == 1)])
		n_a = len(data[(data[attribute] == x) & (data[goal]== 0)])
		total_a = float(p_a + n_a)
		prob = total_a/(p + n)
		q = p_a/total_a
		b = entropy(q)
		r += (prob * b)
	return r

def importance(data,attribute,goal):
	p = len(data[data[goal] == 1])
	n = len(data[data[goal] == 0])
	b = entropy(float(p)/(p+n))
	r = remainder(data,attribute,goal)
	gain = b - r
	return gain