from datamanipulations import get_values_for_col, seperate_classes
from statutils import sumarize_p,summarize_col,normalize_probabilities,pdf

class NaiveBayes:
    def __init__(self):
        self.model = dict()
        self.labels = None
        self.len_of_data_rows = None

    def fit(self, x, y):
        seperated_classes = seperate_classes(x)
        self.summarize_classes(seperated_classes)
        self.labels = y

    def summarize_classes(self,seperated_classes):
        for i in range(len(seperated_classes)):
            self.len_of_data_rows = len(seperated_classes[i+1][0])
            self.model[i+1] = list()
            for j in range(self.len_of_data_rows):
                col = get_values_for_col(j, seperated_classes[i+1])
                mean_val, stddev_val = summarize_col(col)
                self.model[i+1].append((mean_val, stddev_val))

    def calculate_class_probabilities(self,row):
        probabilities = dict()
        for class_value, class_summaries in self.model.items():
            probabilities[class_value] = list()
            for i in range(len(class_summaries)):
                mean, stdev = class_summaries[i]
                probabilities[class_value].append(
                    pdf(row[i], mean, stdev))

        sum_probabilities = sumarize_p(probabilities)

        return normalize_probabilities(probabilities, sum_probabilities)

    def predict(self, rows):
        predictions = list()
        for row in rows:
            probabilities = self.calculate_class_probabilities(row)
            best_label, best_prob = None, -1
            for label, prob in probabilities.items():
                if best_label is None or prob > best_prob:
                    best_prob = prob
                    best_label = label
            predictions.append(best_label)
        return predictions

    def accuracy_score(self, actual, predictions):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predictions[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    def confusion_matrix(self, predicted, actual):
        labels = sorted(set(actual))

        # create matrix by looping through the unique values in labels
        matrix = [[0 for val in labels] for val in labels]

        # set the key for the position in the matrix the value should be updated
        index_map = {key: i for i, key in enumerate(labels)}

        # loop through the predicted and actual values
        for predicted, actual in zip(predicted, actual):
            # get the index from the map of the predicted score
            # increase the value at the position by 1
            matrix[index_map[predicted]][index_map[actual]] += 1
        return matrix

