#!/usr/bin/env python

def nbclassify(alpha):
    with open('vocabulary.txt') as vocabulary_file:
        vocabulary = vocabulary_file.read().split()
    n_vocabulary = len(vocabulary)
    
    with open('newsgrouplabels.txt', 'r') as newsgroup_file:
        newsgroup_names = newsgroup_file.read().split()
    n_newsgroup = len(newsgroup_names)
    
    with open('train.label', 'r') as train_label_file:
        train_labels = [int(x) for x in train_label_file.read().split()]
    n_train_data = len(train_labels)
    
    with open('train.data', 'r') as train_data_file:
        train_data = [map(int, line.split()) for line in train_data_file]
    
    n_y = [0] * n_newsgroup
    for i in range(n_train_data):
        n_y[train_labels[i] - 1] += 1
    p_y = [x / float(n_train_data) for x in n_y]
    
    n_x_given_y = [[alpha] * n_vocabulary for i in range(n_newsgroup)]
    for data in train_data:
        n_x_given_y[train_labels[data[0] - 1] - 1][data[1] - 1] += data[2]
    p_x_given_y = []
    for y in n_x_given_y:
        sum_y = sum(y)
        p_x_given_y.append([px / float(sum_y) for px in y])
    
    n_test_data = 0
    p_y_given_x = []
    with open('test.data') as test_data_file:
        for line in test_data_file:
            current_line = map(int,line.split())
            current_test_number = current_line[0]
            if current_test_number > n_test_data:
                n_test_data += 1
                p_y_given_x.append(list(p_y))
            for j in range(current_line[2]):
                for i in range(len((p_y_given_x[current_test_number -1]))):
                    p_y_given_x[current_test_number - 1][i] *= \
                            p_x_given_y[i][current_line[1] - 1] 
                
                norm_f = sum(p_y_given_x[current_test_number - 1], 0.0) / \
                        len(p_y_given_x[current_test_number - 1])
                p_y_given_x[current_test_number - 1] = [z / norm_f for z in \
                        p_y_given_x[current_test_number - 1]]
                        
    label_classified = [probs.index(max(probs)) + 1 for probs in p_y_given_x]
    
    with open('test.label', 'r') as test_label_file:
        test_labels = [int(x) for x in test_label_file.read().split()]
    confusion_matrix = [[0] * n_newsgroup for i in range(n_newsgroup)]
    n_correct = 0
    for (i,label) in enumerate(test_labels):
        confusion_matrix[label-1][label_classified[i] - 1] += 1
        if label == label_classified[i]:
            n_correct += 1
    accuracy = n_correct / float(n_test_data)
    return [accuracy, confusion_matrix]

if __name__ == '__main__':
    [accuracy, confusion_matrix] = nbclassify(1./61188)
    for row in confusion_matrix:
        print ', '.join(map(str, row))
    print accuracy
