#NOTE: IMPORTING THE REQUIRED MODULES
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import numpy as np

#NOTE: READING THE CSV FILE AND OPERATING ON THE DATASET
df = pd.read_csv('FYP.csv')
X = df.drop(['typeoffault','lineno','t','typeoffault_n','T12VPhaseA','T12VPhaseB',
'T12VPhaseC','T13VPhaseA','T13VPhaseB','T13VPhaseC','T23VPhaseA','T23VPhaseB',
'T23VPhaseC','T24VPhaseA','T24VPhaseB','T24VPhaseC','T25VPhaseA','T25VPhaseB',
'T25VPhaseC','T34VPhaseA','T34VPhaseB','T34VPhaseC','T45VPhaseA','T45VPhaseB',
'T45VPhaseC'],axis='columns')
df.loc[(df.typeoffault_n == 5),'typeoffault_n']=500
df.loc[(df.typeoffault_n == 0),'typeoffault_n']=5
df.loc[(df.typeoffault_n == 500),'typeoffault_n']=0
a=[13,17,19,23,29,31,37]
for i in range(1,8):
    df.loc[(df.lineno==i),'lineno']=a[i-1]
df['linenno_typeoffault_n']=df['typeoffault_n']*df['lineno']
le = LabelEncoder()
targets = pd.DataFrame(data=le.fit_transform(df['linenno_typeoffault_n']))
X = RobustScaler().fit_transform(X)

#NOTE: SPLITTING THE DATASET INTO TESTING AND TRAINING DATASETS
X_train, X_test, y_train, y_test = train_test_split(X, targets, test_size=0.2)
knn = KNeighborsClassifier(n_neighbors=13)
y_train = np.reshape(y_train,(1621620,))
y_test = np.reshape(y_test,(405405,))

#NOTE: TRAINING THE MODEL
knn.fit(X_train, y_train)

#NOTE: PRINTING THE ACCURACY OF THE MODEL
print(knn.score(X_test, y_test))

#NOTE: THE NUMBERING SYSTEM TO IDENTIFY THE TYPE OF FAULT AND THE LINE NUMBER
output={"0": "NO Fault","18":"Line 1 LG Fault","23":"Line 2 LG Fault","27":"Line 3 LG Fault",
"29":"Line 4 LG Fault","32":"Line 5 LG Fault","34":"Line 6 LG Fault","35":"Line 7 LG Fault", 
"1":"Line 1 LL Fault","2":"Line 2 LL Fault","3":"Line 3 LL Fault","4":"Line 4 LL Fault", 
"6":"Line 5 LL Fault","7":"Line 6 LL Fault","9":"Line 7 LL Fault","5":"Line 1 LLG Fault", 
"8":"Line 2 LLG Fault","10":"Line 3 LLG Fault","12":"Line 4 LLG Fault",
"16":"Line 5 LLG Fault", "17":"Line 6 LLG Fault","21":"Line 7 LLG Fault",
"11":"Line 1 LLL Fault","13":"Line 2 LLL Fault","15":"Line 3 LLL Fault",
"20":"Line 4 LLL Fault","24":"Line 5 LLL Fault","26":"Line 6 LLL Fault", 
"28":"Line 7 LLL Fault","14":"Line 1 LLLG Fault","19":"Line 2 LLLG Fault",
"22":"Line 3 LLLG Fault","25":"Line 4 LLLG Fault","30":"Line 5 LLLG Fault",
"31":"Line 6 LLLG Fault","33":"Line 7 LLLG Fault"}

#NOTE: PREDICTING THE TYPE OF FAULT AND THE LINE NUMBER USING THE TEST DATA
y_pred = knn.predict(X_test)
r = pd.DataFrame(y_pred)
for i in r[0].head():
    temp = str(i)
    print(output[temp])
    
#NOTE: CREATING THE CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
# %matplotlib inline
plt.figure(figsize=(15,15))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth') 