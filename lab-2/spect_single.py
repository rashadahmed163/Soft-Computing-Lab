import csv 
import math
import random
from k_fold import fold

def shuffle(array):
	for i in range(len(array)-1,0,-1):
		j=random.randint(0,i+1)
		array[i],array[j]=array[j],array[i]
	return array

def train(train_set,test_set,attributes,constant):
	dataset=[]
	for i in range(len(train_set)):
		test=[]
		for j in range(1,len(attributes)):
			test.append(float(train_set[i][j]))
		dataset.append(test)

	#Assume Iris-setosa=1 and Iris-versicolor=0
	for i in range(len(train_set)):
		if train_set[i][0]=='Yes':
			dataset[i].append(1)
		elif train_set[i][0]=='No':
			dataset[i].append(0)

	#dataset=shuffle(dataset)
	wts=[]
	for i in range(len(attributes)-1):
		wt=random.uniform(-0.5,0.5)
		wts.append(wt)
	#print(dataset)
	y=[0]*len(dataset)
	error=[0]*len(dataset)
	#for k in range(len(dataset)):
	for k in range(500):
		for i in range(len(dataset)):
			summ=0
			for j in range(len(attributes)-1):
				summ+=dataset[i][j]*wts[j]
			if summ>0.0:
				y[i]=1
			elif summ<=0.0:
				y[i]=0
			error[i]=dataset[i][22]-y[i]
			if error[i]!=0:
				for l in range(len(attributes)-1):
					wts[l]=wts[l]+(constant*error[i]*dataset[i][l])
	
	#Testing Part
	for i in range(len(test_set)):
		summ=0
		for j in range(1,len(attributes)):
			summ+=float(test_set[i][j])*wts[j-1]
		if summ>0.0:
			test_set[i].append(1)
		elif summ<=0.0:
			test_set[i].append(0)
	#print(test_set)
	acc=0
	tp=0
	tn=0
	fp=0
	fn=0
	for i in range(len(test_set)):
		if test_set[i][0]=='Yes' and test_set[i][23]==1:
			acc+=1
			tp+=1
		if test_set[i][0]=='No' and test_set[i][23]==0:
			acc+=1
			tn+=1
		if test_set[i][0]=='Yes' and test_set[i][23]==0:
			fn+=1
		if test_set[i][0]=='No' and test_set[i][23]==1:
			fp+=1

	precision=(tp/(tp+fp))*100
	recall=(tp/(tp+fn))*100
	result=(acc/len(test_set))*100
	return [result,precision,recall]


def main():
	filename="SPECT.csv"
	attributes=[]
	rows=[]
	with open(filename,'r') as csvfile:
		csvreader=csv.reader(csvfile)

		attributes=next(csvreader)
		for row in csvreader:
			rows.append(row)

	print(len(attributes))
	k=10
	acc_final=[]
	for i in range(1,k+1):
		constant=i/10
		#constant=0.2
		Rows=shuffle(rows)
		accuracy=[]
		Pre=[]
		Re=[]
		for i in range(1,k+1):
			after_fold=fold(Rows,i,k)
			train_set=after_fold[0]
			test_set=after_fold[1]
			intermediate=train(train_set,test_set,attributes,constant)
			acc=intermediate[0]
			Precision=intermediate[1]
			Recall=intermediate[2]
			accuracy.append(acc)
			Pre.append(Precision)
			Re.append(Recall)
		#print(accuracy)
		acc_sum=0
		pre_sum=0
		re_sum=0
		for i in accuracy:
			print(i)
		for i in accuracy:
			acc_sum+=i
		for i in Pre:
			pre_sum+=i
		for i in accuracy:
			re_sum+=i
		acc_final.append(acc_sum/k)
		print("For Learning Rate : ",constant," Accuracy : ",acc_sum/k)
		print("Precision : ",pre_sum/k," Recall : ",re_sum/k)
		
	

if __name__ == '__main__':
	main()