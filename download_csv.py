path_to_data = 'train.csv'
path_to_new = 'positives.csv'

import csv

with open(path_to_data) as train_data, open(path_to_new, 'w') as pos_data:
    reader = csv.DictReader(train_data)
    writer = csv.writer(pos_data)
    for row in reader:
        if row['is_attributed'] == '1':
            writer.writerow(row.values())
