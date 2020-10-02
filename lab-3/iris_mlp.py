import csv 
import math
import random
from k_fold import fold

def shuffle(array):
	for i in range(len(array)-1,0,-1):
		j=random.randint(0,i)
		array[i],array[j]=array[j],array[i]
	return array

def sigmoid(x):
	x = x*-1
	sig = 1/(1+math.exp(x))
	return sig

def train(train_set, test_set, attributes, constant, n, threshold):
	#Replicating train_set to train_data to convert sting emtries to float entries
	train_data = []
	for i in range(len(train_set)):
		test = []
		for j in range(len(attributes) - 1):
			test.append(float(train_set[i][j]))
		train_data.append(test)

	for i in range(len(train_set)):
		train_data[i].append(train_set[i][4])

	#Initializing random weights to the edges between input and hidden layer
	wts_ij = []
	for i in range(len(attributes)-1):
		test_wt = []
		for j in range(n):
			wt = random.uniform(-0.5, 0.5)
			test_wt.append(wt)
		wts_ij.append(test_wt)

	#Initializing random weights to the edges in between hidden and output layer
	wts_jk = []
	for i in range(n):
		wt = random.uniform(-0.9, 0.9)
		wts_jk.append(wt)

	Ij = [0 for i in range(n+1)] #Input to the hidden and output layer
	Oj = [0 for i in range(n+1)] #Output from the hidden and output layer
	errj = [0 for i in range(n+1)] #Error at hidden and output layer
	
	for k in range(500):
		for x in train_data:
			if x[4] == "Iris-setosa":
				Tj = 1 #Target output
			elif x[4] == "Iris-versicolor":
				Tj = 0
			#Calculating input at hidden layer
			for j in range(n):
				summ = 0
				for i in range(len(attributes)-1):
					summ += wts_ij[i][j]*x[i]
				Ij[j] = summ
				#Calculating output at hidden layer using sigmoid function
				Oj[j] = sigmoid(Ij[j])
			
			#Calculating input at ouput layer
			k = 0
			for j in range(n):
				k += wts_jk[j]*Oj[j]
			Ij[5] = k

			#Calculating output at output layer
			Oj[5] = sigmoid(Ij[5])

			#Calculating error at output layer
			errj[5] = Oj[5]*(1-Oj[5])*(Tj-Oj[5])

			#Back-propagate the error from output layer ton the hidden layer
			for j in range(n):
				errj[j] = Oj[j]*(1-Oj[j])*wts_jk[j]*errj[5]

			#Updating weights of the edges between the input and output layer
			for i in range(len(attributes)-1):
				for j in range(n):
					wts_ij[i][j] = wts_ij[i][j]+(errj[j]*x[i]*constant)

			#Updating weights of the edges between the hidden and output layer
			for j in range(n):
				wts_jk[j] = wts_jk[j]+(errj[5]*Oj[j]*constant)

	#Testing Part
	test_data = []
	for i in range(len(test_set)):
		testt = []
		for j in range(len(attributes) - 1):
			testt.append(float(test_set[i][j]))
		test_data.append(testt)

	for i in range(len(test_set)):
		test_data[i].append(test_set[i][4])


	for k in range(len(test_data)):
		su = 0
		summ = [0 for i in range(n)]
		for i in range(n):
			for j in range(len(attributes) - 1):
				summ[i] += wts_ij[j][i]*float(test_data[k][j])

		for l in range(n):
			su += sigmoid(summ[l])*wts_jk[l]

		if sigmoid(su) > threshold:
			test_data[k].append(1)
		elif sigmoid(su) <= threshold:
			test_data[k].append(0)

	#Calculating Accuracy, Precision and Recall
	acc, tp, tn, fp, fn = 0, 1, 1, 1, 1
	for i in range(len(test_set)):
		if test_data[i][4] == 'Iris-setosa' and test_data[i][5] == 1:
			acc += 1
			tp += 1
		elif test_data[i][4] == 'Iris-versicolor' and test_data[i][5] == 0:
			tn += 1
			acc += 1
		elif test_data[i][4] == 'Iris-setosa' and test_data[i][5] == 0:
			fn += 1 
		elif test_data[i][4] == 'Iris-versicolor' and test_data[i][5] == 1:
			fp += 1

	precision = (tp/(tp+fp))*100
	recall = (tp/(tp+fn))*100
	result = (acc/len(test_set))*100
	return [result,precision,recall]



def main():
	filename = "IRIS.csv"
	attributes = []
	rows = []
	with open(filename,'r') as csvfile:
		csvreader = csv.reader(csvfile)
		attributes = next(csvreader)
		for row in csvreader:
			rows.append(row)
	print(len(attributes))
	k = 10
	#n=int(input("Enter the number of nodes at hidden layer : "))
	n=5
	max_acc=0
	threshold = 0.1
	for j in range(k):
		for i in range(1, k+1):
			constant = i/10
			Rows = shuffle(rows)
			accuracy = []
			Pre = []
			Re = []
			for i in range(1, k+1):
				after_fold = fold(Rows,i,k)
				train_set = after_fold[0]
				test_set = after_fold[1]
				intermediate = train(train_set,test_set,attributes,constant,n,threshold)
				acc = intermediate[0]
				Precision = intermediate[1]
				Recall = intermediate[2]
				accuracy.append(acc)
				Pre.append(Precision)
				Re.append(Recall)
			acc_sum, pre_sum, re_sum =0, 0, 0
			for i in accuracy:
				print(i)
			for i in accuracy:
				acc_sum += i
			for i in Pre:
				pre_sum += i
			for i in accuracy:
				re_sum += i
			if (acc_sum/k) > max_acc:
				max_acc = acc_sum/k
				max_constant = constant
			print("For Learning Rate : ",constant," Accuracy : ",acc_sum/k)
			print("Precision : ",pre_sum/k," Recall : ",re_sum/k)
			print("--------------------------------------------")
		print("Maximum Accuracy : ",max_acc," For learning rate : ",max_constant," For Threshold : ",threshold)
		threshold += 0.1
		

if __name__ == '__main__':
	main()
