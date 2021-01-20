from naivebayes import NaiveBayes
import csv_parser
iris = "./iris.csv"
notes = "./banknote_authentication.csv"
iris_list, iris_labels = csv_parser.parse_data(iris)
note_list, note_labels = csv_parser.parse_data(notes)
bayes = NaiveBayes()


bayes.fit(note_list, note_labels)
note_predictions = bayes.predict(note_list)
note_accuracy = bayes.accuracy_score(note_labels, note_predictions)
note_confusion_matrix = bayes.confusion_matrix(note_predictions,note_labels)
print(f"Note accuracy: {note_accuracy}%")
print(note_confusion_matrix)

bayes.fit(iris_list, iris_labels)
iris_predictions = bayes.predict(iris_list)
iris_accuracy = bayes.accuracy_score(iris_labels, iris_predictions)
print(f"Iris accuracy: {iris_accuracy}%")
iris_confusion_matrix = bayes.confusion_matrix(iris_predictions, iris_labels)
print(iris_confusion_matrix)
