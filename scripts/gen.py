import numpy as np

class_names = np.loadtxt('./data/classnames.txt', delimiter='\n', dtype=np.str)

with open('./models/tf_api_models/object-detection.pbtxt', 'a') as file:
    for i in range(len(class_names)):
        file.write('item {\n')
        file.write(f'\tid: {i}\n')
        file.write(f'\tname: {class_names[i]}\n')
        file.write('}\n\n')
