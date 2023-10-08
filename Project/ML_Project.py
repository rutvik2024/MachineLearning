#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[53]:


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,classification_report
import numpy as np
import os
import cv2


# In[54]:


file_path = r'/home/rutvik/Desktop/ML Project/Project_M22CS011_M22CS052_M22CS059/Dataset'
classes = ['biodegradable','glass','metal','paper','plastic']
IMG_SIZE=60


# ## Testing for Image Read

# In[55]:
print("Image Reading")

for waste_class in classes:
    path=os.path.join(file_path, waste_class)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img))
        new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
        new_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        plt.imshow(new_array)
        plt.show()
        break
    break

print("Image Reading Done..!!")
# In[56]:


training_data=[]
data_count = []
def create_training_data():
    for waste_class in classes:
        path=os.path.join(file_path, waste_class)
        class_num=classes.index(waste_class)
        count = 0
        for img in os.listdir(path):
            count = count + 1
            try:
                img_array=cv2.imread(os.path.join(path,img))
                new_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                new_array=cv2.resize(new_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
            
        data_count.append(count)
create_training_data()  


# In[57]:


print(data_count)
total_data = len(training_data)
print(total_data)

# Percentage of data for each class
per_data = []
for i in range(5):
    count = data_count[i]/float(total_data)
    count = round(count,2)
    per_data.append(count*100)
print(per_data)


# ## plot dataset on pie chart

# In[58]:


fig1, ax1 = plt.subplots()
explode = (0, 0.1, 0, 0, 0) 
ax1.pie(per_data, explode=explode, labels=classes,autopct='%1.1f%%',
        shadow=True, startangle=90)
 
plt.show()


# In[59]:


lenofimage = len(training_data)
X=[]
y=[]

for categories, label in training_data:
    X.append(categories)
    y.append(label)
X= np.array(X).reshape(lenofimage,-1)



# ## Data Scaling 

# In[60]:


X = X/255.0


# In[61]:


y=np.array(y)


# ## Data Split into Training and Testing

# In[62]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25)


# 1) model import
# 2) train
# 3) testing
# 4) accuracy print


# ## Support Vector Machine

# In[63]:


from sklearn import svm
model = svm.SVC()
model.fit(X_train, y_train) # training
# pred = model.predict(X_train) # test on training data
pred_test = model.predict(X_test) # 
# print(accuracy_score(pred, y_train))
print(accuracy_score(pred_test, y_test))


# ### Model Predication Using SVM

# In[7]:


def model_svm_pred(img):
    pred_test = model.predict(img)
    
    if(pred_test[0] == 0):
        print("Image belong to Biodegradable")
        
    elif(pred_test[0] == 1):
        print("Image belong to glass")
        
    elif(pred_test[0] == 2):
        print("Image belong to metal")
        
    elif(pred_test[0] == 3):
        print("Image belong to paper")
        
    elif(pred_test[0] == 4):
        print("Image belong to plastic")
    


# In[65]:


img = r'/home/rutvik/Desktop/ML Project/Project/Dataset/glass/gl_0.jpg'
img_array=cv2.imread(img)
new_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
new_array=cv2.resize(new_array,(IMG_SIZE,IMG_SIZE))
new_array= np.array(new_array).reshape(1,-1)
model_svm_pred(new_array)


# ### SVM Classification Report

# In[66]:

# Confusion matrix
print(classification_report(y_test, pred_test))


# ### SVM Accuracy Score :

# In[67]:


acc_svc = round(model.score(X_test, y_test) * 100, 2)
acc_svc


# ## K-Nearest Neighbour

# In[68]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
pred_test = knn.predict(X_test)
# print('Prediction : {}'.format(pred_test))
print('Accuracy with KNN : ',knn.score(X_test,y_test))
# print(X_test)
# print(type(pred_test[0]))


# ### Model Predication

# In[69]:


def model_knn_pred(img):
    pred_test = knn.predict(img)
    
    if(pred_test[0] == 0):
        print("Image belong to Biodegradable")
        
    elif(pred_test[0] == 1):
        print("Image belong to glass")
        
    elif(pred_test[0] == 2):
        print("Image belong to metal")
        
    elif(pred_test[0] == 3):
        print("Image belong to paper")
        
    elif(pred_test[0] == 4):
        print("Image belong to plastic")

model_knn_pred(img)

# ### KNN Classification Report

# In[72]:


print(classification_report(y_test, pred_test))


# ### KNN Accuracy Score :

# In[73]:


acc_knn = round(knn.score(X_test, y_test) * 100, 2)
acc_knn


# ## Decision Tree

# In[74]:


from sklearn.tree import DecisionTreeClassifier
dcsTree = DecisionTreeClassifier()
pred = dcsTree.fit(X_train, y_train)
pred_test = pred.predict(X_test)
print("Test Set Accuracy with Decision tree : {}".format(accuracy_score(y_test, pred_test)))


# ### Decision Tree Classification Report

# In[76]:


print(classification_report(y_test, pred_test))


# ### Decision Tree Accuracy Score :

# In[77]:


acc_decision_tree = round(dcsTree.score(X_test, y_test) * 100, 2)
acc_decision_tree


# ## Model Prediction Algorithm for Decision Tree

# In[78]:


def model_dt_pred(img):
    pred_test = pred.predict(img)
    
    if(pred_test[0] == 0):
        print("Image belong to Biodegradable")
        
    elif(pred_test[0] == 1):
        print("Image belong to glass")
        
    elif(pred_test[0] == 2):
        print("Image belong to metal")
        
    elif(pred_test[0] == 3):
        print("Image belong to paper")
        
    elif(pred_test[0] == 4):
        print("Image belong to plastic")


# ## Decision Tree Representation

# In[79]:


plt.figure(figsize=(12,8))

from sklearn import tree

tree.plot_tree(dcsTree.fit(X_train, y_train)) 


# In[80]:


import graphviz 
dot_data = tree.export_graphviz(clf_gini, out_file=None, 
                              feature_names=X_train.columns,  
                              class_names=y_train,  
                              filled=True, rounded=True,  
                              special_characters=True)

graph = graphviz.Source(dot_data) 

graph 


# ## Random Forest

# In[81]:


from sklearn.ensemble import RandomForestClassifier
rnd_forest = RandomForestClassifier(n_estimators=5, random_state=0)
pred = rnd_forest.fit(X_train, y_train)
pred_test = pred.predict(X_test)
print("Accuracy with Random Forest : {}".format(accuracy_score(y_test, pred_test)))


# ## Model Prediction Using Random Forest

# In[82]:


def model_rnd_pred(img):
    pred_test = pred.predict(img)
    
    if(pred_test[0] == 0):
        print("Image belong to Biodegradable")
        
    elif(pred_test[0] == 1):
        print("Image belong to glass")
        
    elif(pred_test[0] == 2):
        print("Image belong to metal")
        
    elif(pred_test[0] == 3):
        print("Image belong to paper")
        
    elif(pred_test[0] == 4):
        print("Image belong to plastic")


# ### Random Forest Classification report

# In[83]:


print(classification_report(y_test, pred_test))


# ### Random Forest Accuracy Score :

# In[84]:


acc_random_forest = round(rnd_forest.score(X_test, y_test) * 100, 2)
acc_random_forest


# ## Model Prediction for given data image using different machine learning algorithm

# In[85]:


def pred_image(new_array):
    print("Using SVM : ")
    model_svm_pred(new_array)
    print("------------------------------------------------")

    print("Using KNN : ")
    model_knn_pred(new_array)
    print("------------------------------------------------")

    print("Using Decision Tree : ")
    model_dt_pred(new_array)
    print("------------------------------------------------")

    print("Using Random Forest : ")
    model_rnd_pred(new_array)


# In[88]:


img = r'/home/rutvik/Desktop/ML Project/Project/Dataset/glass/gl_6.jpg'
img_array=cv2.imread(img)
# new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
new_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
new_array=cv2.resize(new_array,(IMG_SIZE,IMG_SIZE))
new_array= np.array(new_array).reshape(1,-1)


pred_image(new_array)


# ## Model Summary

# In[89]:


models = ('SVM', 'KNN', 'Decision Tree', 'Random Forest')
x_pos = np.arange(len(models))
acc = [acc_svc,acc_knn,acc_decision_tree,acc_random_forest]
plt.bar(x_pos, acc, align='center', alpha=0.5, color='r')
plt.xticks(x_pos, models, rotation='vertical')
plt.ylabel('Accuracy')
plt.title('Classifier Outcome')
plt.show()


# In[ ]:




