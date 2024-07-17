import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
import sklearn.metrics as mt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
df = pd.read_csv("titanic.csv")
q=['PassengerId','Name','Ticket','Cabin','SibSp','Parch','Age']
df.drop(q,axis=1,inplace=True) #q listesindeki kolonları sildik

df['Embarked']=df['Embarked'].fillna('S') #Embarked kolonundaki eksik verileri S ile doldurduk

mean=df['Fare'].mean()
df['Fare']=df['Fare'].fillna(mean) #Fare kolonundaki eksik verileri ortalama ile doldurduk

labelencoder=LabelEncoder()
df['Sex']=df.iloc[:,2]=labelencoder.fit_transform(df.iloc[:,2].values) #Sex kolonunu sayısal veriye dönüştürdük
df['Embarked']=df.iloc[:,4]=labelencoder.fit_transform(df.iloc[:,4].values) #Embarked kolonunu sayısal veriye dönüştürdük  

object=StandardScaler()
object.fit_transform(df)  #verileri standartlaştırdık  


x = df.iloc[:,1:5].values #bağımsız değişkenler
y = df.iloc[:,0].values #bağımlı değişken

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4,random_state=4) #verileri train ve test olarak ayırdık

accuracy_knn=[]
p_knn=[]
r_knn=[]
f_knn=[]

knn_komsular=[3,7,11] #knn için komşu sayılarını sizin istediğiniz gibi belirledik
for i in knn_komsular:
    knn_model=KNeighborsRegressor(n_neighbors=i).fit(x_train,y_train)
    y_pred = knn_model.predict(x_test)
    tn,fp,fn,tp=mt.confusion_matrix(y_test,y_pred.round()).ravel()
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    f_score=2*precision*recall/(precision+recall)
    accuracy_knn.append(accuracy)
    p_knn.append(precision)
    r_knn.append(recall)
    f_knn.append(f_score)

#print(accuracy_knn)
#print(p_knn)
#print(r_knn)
#print(f_knn) 


mlp_accuracy=[]
mlp_p=[]
mlp_r=[]
mlp_f=[]
mlp_sayilar=[1,2,3] #mlp için katman sayılarını sizin istediğiniz gibi belirledik

for i in mlp_sayilar :
    mlp_model=MLPRegressor(alpha=i,hidden_layer_sizes=(32,32,32),max_iter=1000).fit(x_train,y_train)
    y_pred = mlp_model.predict(x_test)
    tn,fp,fn,tp=mt.confusion_matrix(y_test,y_pred.round()).ravel()
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    f_score=2*precision*recall/(precision+recall)
    mlp_accuracy.append(accuracy)
    mlp_p.append(precision)
    mlp_r.append(recall)
    mlp_f.append(f_score)

#print(mlp_accuracy)
#print(mlp_p)
#print(mlp_r)
#print(mlp_f)
    


nb_model=GaussianNB().fit(x_train,y_train)
y_pred = nb_model.predict(x_test)
tn,fp,fn,tp=mt.confusion_matrix(y_test,y_pred.round()).ravel()
accuracy=(tp+tn)/(tp+tn+fp+fn)
precision=tp/(tp+fp)
recall=tp/(tp+fn)
f_score=2*precision*recall/(precision+recall)
print(accuracy)
print(precision)
print(recall)
print(f_score)



