import train_test
import pandas as pd
#import class_diagnostics
import GLM_Bino

#print(dir(unittest.TestCase))

X_test = train_test.X_test 
X_train = train_test.X_train
#X_train = 2
Y_test = train_test.Y_test
Y_train = train_test.Y_train.to_frame()
#Y_train = 2
threshold = 0.47
function = GLM_Bino.GLM_Binomial_fit

print(type(X_test["CHILDREN"]))
print(X_test["CHILDREN"].name)