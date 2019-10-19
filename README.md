## 1，Introduction

The easyeda is a simple but useful tool to do Exploratory Data Analysis in Machine Learning.
It can be used in both classification task and regression task. 


## 2，Use Example

First,you can use pip to install easyeda.

```bash
pip install easyeda
```

Then, you can use it like below.

```python
from easyeda import eda
import pandas ad pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

## 
boston = datasets.load_boston()
df = pd.DataFrame(boston.data,columns = boston.feature_names)
df["label"] = boston.target
dftrain,dftest = train_test_split(df,test_size = 0.3)
dfeda = eda(dftrain,dftest,language="Chinese")

```

### 3，Contact to the author

Github: https://github.com/lyhue1991/easyeda

Email:lyhue1991@163.com

