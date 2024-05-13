import os
import re
import csv

use_dir = "metrics/leave-one-out/USE/gemini/pretrained/"
meteor_dir = "metrics/leave-one-out/METEOR/gemini/pretrained/"
filename = "metrics/leave-one-out/jam-pretrained-gemini.csv"

arr = os.listdir(use_dir)
usefiles = []
meteorfiles = []
for file in arr:
    if file.endswith('.txt'):
        usefiles.append(use_dir + file)

for file in arr:
    if file.endswith('.txt'):
        meteorfiles.append(meteor_dir + file)

final_use = {}
final_meteor = {}

for usefile in usefiles:
    holdoutnumber = usefile.split(f"{use_dir}")[-1]
    holdoutnumber = holdoutnumber.split(".txt")[0]
    holdoutnumber = re.findall(r'\d+', holdoutnumber)[0]
    total = 0
    usedat = open(usefile, 'r')
    count = 0
    for c, line in enumerate(usedat):
        if(line.startswith("Traceback")):
            continue
        else:
            score = line.split("is")[-1]
            score = score.split("\n")[0]
            score = float(score)
            total += score
            count += 1
    avg = total / count
    final_use[holdoutnumber] = avg

for meteorfile in meteorfiles:
    holdoutnumber = meteorfile.split(f"{meteor_dir}")[-1]
    holdoutnumber = holdoutnumber.split(".txt")[0]
    holdoutnumber = re.findall(r'\d+', holdoutnumber)[0]
    total = 0
    meteordat = open(meteorfile, 'r')
    count = 0
    for c, line in enumerate(meteordat):
        if("FileNotFoundError" in line):
            continue
        else:
            score = line.split("final status for")[-1]
            score = score.split("M")[-1]
            score = score.split("\n")[0]
            score = float(score)
            total += score
            count += 1
    avg = total / count
    final_meteor[holdoutnumber] = avg

fields = ["holdout_method", "meteor", "use"]

with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(fields)
    for i in range(0, len(list(final_meteor.keys()))):
        meteor = str(final_meteor[str(i)])
        use = str(final_use[str(i)])
        final = [str(i), meteor, use]
        csvwriter.writerow(final)




