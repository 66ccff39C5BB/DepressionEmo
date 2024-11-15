import csv
label_names = ["loneliness", "hopelessness", "sadness", "brain dysfunction (forget)", "worthlessness", "emptiness", "anger", "suicide intent"]

for i in range(8):
    keywords1 = {}
    keywords2 = {}
    with open('keywords/'+ label_names[i]+'_true.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            keywords1[row[0]] = float(row[1])
    with open('keywords/'+ label_names[i]+'_pred.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            keywords2[row[0]] = float(row[1])
    keywords = []
    for k,v in keywords1.items():
        if k in keywords2:
            keywords.append((k,'{:.2f}%'.format((keywords2[k]-v)/v*100)))

    with open('keywords/'+ label_names[i]+'_minus.csv', 'w+') as f:
        csvWriter = csv.writer(f)
        for row in keywords: csvWriter.writerow(row)
    print(label_names[i])