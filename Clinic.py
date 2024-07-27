from joblib import load
import numpy as np
import pandas as pd
from xgboost import plot_tree
import matplotlib.pyplot as pt




doctor = load('Doctor.joblib')
label_encoder = load('Encoder.joblib')
X,x,Y,y = load('DataSet.joblib')



def check(*position_1):
    mask = list(np.zeros(len(X.columns)))
    for pos in position_1:
        mask[X.columns.get_loc(pos)] = 1
    inp = pd.DataFrame([mask],dtype=int, columns=X.columns)
    return inp


inp_mask = check("fatigue", "weight_loss", "lethargy","high_fever", "sweating", "diarrhoea", "receiving_blood_transfusion", "receiving_unsterile_injections","muscle_pain")

report = doctor.predict(inp_mask)
print(label_encoder.inverse_transform(report))



plot_tree(doctor, num_trees=0)

pt.title('Tree Visualization')
pt.show()



