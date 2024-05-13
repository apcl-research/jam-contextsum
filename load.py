import pickle

alldata = pickle.load(open("/nfs/projects/EyeContext/summaries/eyecontext_summaries.pkl", "rb"))

p7 = []
p6 = []
for data in alldata:
    if(data['participant'] == 'P7'):
        p7.append(data)
    elif(data['participant'] == 'P6'):
        p6.append(data)
print(len(p7))
print(len(p6))

#print(data)
