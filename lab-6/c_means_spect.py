import csv 
import math
import random
import copy

def shuffle(array):
	for i in range(len(array)-1,0,-1):
		j=random.randint(0,i)
		array[i],array[j]=array[j],array[i]
	return array

def Euclid_distance(x,y):
	dist=0
	for i in range(len(x)):
		dist+=pow((x[i]-y[i]),2)
	dist=math.sqrt(dist)
	return dist

def c_mean_cluster(array,attributes,c,m):
	yes=0
	no=0
	for i in range(len(array)):
		if array[i][0]=="Yes":
			yes+=1
		elif array[i][0]=="No":
			no+=1
	print("Actual Yes : ",yes)
	print("Actual No : ",no)

	dataset=[]
	original_label=[]
	for i in range(len(array)):
		test=[]
		for j in range(1,len(attributes)):
			test.append(float(array[i][j]))
		dataset.append(test)
		original_label.append(array[i][0])
	#print(dataset)

	#Step1:- Choose random centroids
	cluster_centre=[]
	a,b=random.sample(range(0,len(dataset)-1),2)
	#print(a,b)
	cluster_centre.append(dataset[a])
	cluster_centre.append(dataset[b])

	epoch=0
	iterations=100
	while True:
		#Step2:- Compute membership matrix
		membership_value=[[0 for i in range(c)] for j in range(len(dataset))]
		for i in range(len(dataset)):
			for j in range(c):
				num=Euclid_distance(dataset[i],cluster_centre[j])
				if num==0:
					membership_value[i][j]=1
					break
				#print(num)
				den=0
				Den=0
				for l in range(c):
						den=Euclid_distance(dataset[i],cluster_centre[l])
						if den==0:
							membership_value[i][j]=1
							break
						Den+=math.pow(num/den,2/(m-1))
						#print(Den)
						if Den==0:
							membership_value[i][j]=1
						else:
							membership_value[i][j]=1/Den
		#print(membership_value)

		#Step3:- Compute Cluster centre
		new_cluster_centre=[]
		for i in range(c):
			test=[]
			for j in range(len(dataset[0])):
				sum_num=0
				sum_den=0
				for l in range(len(dataset)):
					sum_num+=(membership_value[l][i]**m)*dataset[l][j]
					sum_den+=(membership_value[l][i]**m)
				if sum_num==0 or sum_den==0:
					test.append(0)
				else:
					test.append(sum_num/sum_den)
			new_cluster_centre.append(test)
		#print(new_cluster_centre)

		if new_cluster_centre==cluster_centre or iterations==0:
			break
		else:
			cluster_centre=copy.deepcopy(new_cluster_centre)
		#print(cluster_centre)
		epoch+=1
		iterations-=1
	print("Number of epoch : ",epoch)
	predicted_label=[]
	#print(cluster_centre)

	pre_yes=0
	pre_no=0
	pre_yno=0
	for j in range(len(dataset)):
		if membership_value[j][0]>membership_value[j][1]:
			predicted_label.append("Yes")
			pre_yes+=1
		elif membership_value[j][0]<membership_value[j][1]:
			predicted_label.append("No")
			pre_no+=1
		else:
			predicted_label.append("Yno")
			pre_yno+=1
	
	print("Predicted Yes : ",pre_yes,"Predicted No : ",pre_no,"Predicted Yno : ",pre_yno)
	acc=0
	for i in range(len(dataset)):
		if original_label[i]==predicted_label[i] or predicted_label[i]=="Yno":
			acc+=1
	print("Count : ",acc)
	acc=acc/len(dataset)
	if pre_yes>pre_no:
		print("Accuracy : ",acc*100)
	else:
		print("Accuracy : ",(1-acc)*100)

def main():
	filename="SPECT.csv"
	attributes=[]
	rows=[]
	with open(filename,'r') as csvfile:
		csvreader=csv.reader(csvfile)

		attributes=next(csvreader)
		for row in csvreader:
			rows.append(row)
	Rows=shuffle(rows)
	#Rows=list(rows)

	c=2
	m=2
	c_mean_cluster(Rows,attributes,c,m)

if __name__ == '__main__':
	main()
