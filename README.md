## 1, Introduction

The edakit is a simple but useful tool to do Exploratory Data Analysis in Machine Learning.
It can be used in both classification task and regression task. 


## 2，Use Example

```python 
from edakit import eda
from sklearn import datasets
from sklearn.model_selection import train_test_split

## 
boston = datasets.load_boston()
df = pd.DataFrame(boston.data,columns = boston.feature_names)
df["label"] = boston.target
dfdata = df.copy()
dftrain,dftest = train_test_split(df,test_size = 0.3)
dfeda = eda(dftrain,dftest,language="Chinese")

```