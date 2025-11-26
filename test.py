import pickle

# open pkl file
with open('./data/cn/equity/data/2021-current/turnover_amounts.pkl', 'rb') as f:
    data = pickle.load(f)  
print(data)