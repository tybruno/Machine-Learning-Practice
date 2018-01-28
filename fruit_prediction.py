from sklearn import tree

#**********sample data **************
# weight  table    label
# 150g    bumpy    orange
# 170g    bumpy    orange
# 140g    smooth   Apple
# 130g    smooth   Apple

#build from the sample table data
features = [[140,1],[130,1],[150,0],[170,0]]
labels = ["apple", "apple","orange","orange"]

clf = tree.DecisionTreeClassifier() #the type of classifier for supervised learning problem

clf = clf.fit(features,labels) #The learning algorithm (finds patterns in data)

#160 is the weight and 0 is bumpyness of the unknown fruit
print (clf.predict([[160,0]])) #Hypothesis: it is an orange because it is bumpy and it's heavy
