import json
import numpy as np
import matplotlib.pyplot as plt

def label_transform(label_id, label_list_length):
    data_label = str(label_id)
    while len(data_label) < label_list_length:
        data_label = '0' + data_label
    return [int(data_label[i]) for i in range(len(data_label))]

def read_labels(datapath, label_list_length):
    # Read the labels from the file
    labels = []
    with open(datapath, 'r') as file:
        for line in file:
            data = json.loads(line)
            label = data["label_id"]
            labels.append(label_transform(label, label_list_length))
    return labels

def pearson_correlation_coefficient(labels):
    # Calculate Pearson correlation coefficient for each pair of labels
    num_labels = labels.shape[0]
    corr_matrix = np.zeros((num_labels, num_labels))
    
    # Compute the Pearson correlation for each pair of labels
    for i in range(num_labels):
        for j in range(i, num_labels):
            # Compute the correlation between label i and label j
            mean_i = np.mean(labels[i])
            mean_j = np.mean(labels[j])
            numerator = np.sum((labels[i] - mean_i) * (labels[j] - mean_j))
            denominator = np.sqrt(np.sum((labels[i] - mean_i)**2) * np.sum((labels[j] - mean_j)**2))
            
            if denominator != 0:
                corr_matrix[i, j] = numerator / denominator
                corr_matrix[j, i] = corr_matrix[i, j]  # Symmetric matrix
            else:
                corr_matrix[i, j] = corr_matrix[j, i] = 0  # No correlation if denominator is zero
                
    return corr_matrix

def save_correlation_matrix_as_figure(corr_matrix, labels, savename='correlation_matrix.png'):
    # Display the correlation matrix
    plt.figure(figsize=(len(labels), len(labels)))
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.title('Correlation Matrix')
    plt.savefig(savename)


if __name__ == '__main__':
    emotion_list = ['anger', 'brain dysfunction (forget)', 'emptiness', 'hopelessness', 'loneliness', 'sadness', 'suicide intent', 'worthlessness']
    symptom_list = ['Sadness', 'Pessimism', 'Sense_of_failure', 'Loss_of_Pleasure', 'Guilty_feelings', 'Sense_of_punishment', 'Self-dislike', 'Self-incrimination', 'Suicidal_ideas', 'Crying', 'Agitation', 'Social_withdrawal', 'Indecision', 'Feelings_of_worthlessness', 'Loss_of_energy', 'Change_of_sleep', 'Irritability', 'Changes_in_appetite', 'Concentration_difficulty', 'Tiredness_or_fatigue', 'Loss_of_interest_in_sex']
    emotion_train_labels = read_labels('Dataset/train.json', len(emotion_list))
    emotion_val_labels = read_labels('Dataset/val.json', len(emotion_list))
    emotion_test_labels = read_labels('Dataset/test.json', len(emotion_list))
    emotion_labels = np.array(emotion_train_labels + emotion_val_labels + emotion_test_labels)
    emotion_corr = pearson_correlation_coefficient(emotion_labels.T)
    save_correlation_matrix_as_figure(emotion_corr, emotion_list, "emotion_correlation_matrix.png")
    
    symptom_train_labels = read_labels('Dataset/train_BDISen.json', len(symptom_list))
    symptom_val_labels = read_labels('Dataset/val_BDISen.json', len(symptom_list))
    symptom_test_labels = read_labels('Dataset/test_BDISen.json', len(symptom_list))
    symptom_labels = np.array(symptom_train_labels + symptom_val_labels + symptom_test_labels)
    symptom_corr = pearson_correlation_coefficient(symptom_labels.T)
    save_correlation_matrix_as_figure(symptom_corr, symptom_list, "symptom_correlation_matrix.png")