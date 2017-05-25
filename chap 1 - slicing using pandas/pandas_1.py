# Import the Pandas library
import pandas as pd
# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

print(train.describe())
print(test.describe())

print(train.shape)
print(train.shape)

<script.py> output:
           PassengerId    Survived      Pclass         Age       SibSp  \
    count   891.000000  891.000000  891.000000  714.000000  891.000000   
    mean    446.000000    0.383838    2.308642   29.699118    0.523008   
    std     257.353842    0.486592    0.836071   14.526497    1.102743   
    min       1.000000    0.000000    1.000000    0.420000    0.000000   
    25%     223.500000    0.000000    2.000000   20.125000    0.000000   
    50%     446.000000    0.000000    3.000000   28.000000    0.000000   
    75%     668.500000    1.000000    3.000000   38.000000    1.000000   
    max     891.000000    1.000000    3.000000   80.000000    8.000000   
    
                Parch        Fare  
    count  891.000000  891.000000  
    mean     0.381594   32.204208  
    std      0.806057   49.693429  
    min      0.000000    0.000000  
    25%      0.000000    7.910400  
    50%      0.000000   14.454200  
    75%      0.000000   31.000000  
    max      6.000000  512.329200  
           PassengerId      Pclass         Age       SibSp       Parch        Fare
    count   418.000000  418.000000  332.000000  418.000000  418.000000  417.000000
    mean   1100.500000    2.265550   30.272590    0.447368    0.392344   35.627188
    std     120.810458    0.841838   14.181209    0.896760    0.981429   55.907576
    min     892.000000    1.000000    0.170000    0.000000    0.000000    0.000000
    25%     996.250000    1.000000   21.000000    0.000000    0.000000    7.895800
    50%    1100.500000    3.000000   27.000000    0.000000    0.000000   14.454200
    75%    1204.750000    3.000000   39.000000    1.000000    0.000000   31.500000
    max    1309.000000    3.000000   76.000000    8.000000    9.000000  512.329200
    (891, 12)
    (891, 12)