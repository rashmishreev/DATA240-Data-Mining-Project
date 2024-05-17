# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def feature_engineering(self):
        self.data['Received_Packet_Rate'] = self.data['Received Packets'] / self.data['Port alive Duration (S)']
        self.data['Sent_Packet_Rate'] = self.data['Sent Packets'] / self.data['Port alive Duration (S)']
        self.data['Received_Byte_Rate'] = self.data['Received Bytes'] / self.data['Port alive Duration (S)']
        self.data['Sent_Byte_Rate'] = self.data['Sent Bytes'] / self.data['Port alive Duration (S)']
        self.data['Received_Error_Rate'] = self.data['Packets Rx Errors'] / self.data['Received Packets']
        self.data['Sent_Error_Rate'] = self.data['Packets Tx Errors'] / self.data['Sent Packets']
        self.data['Delta_Received_Packets'] = self.data['Received Packets'] - self.data['Delta Received Packets']
        self.data['Delta_Received_Bytes'] = self.data['Received Bytes'] - self.data['Delta Received Bytes']
        self.data['Delta_Sent_Bytes'] = self.data['Sent Bytes'] - self.data['Delta Sent Bytes']
        self.data['Delta_Sent_Packets'] = self.data['Sent Packets'] - self.data['Delta Sent Packets']
        self.data['Delta_Port_Alive_Duration'] = self.data['Port alive Duration (S)'] - self.data['Delta Port alive Duration (S)']
        self.data['Switch ID'] = self.data['Switch ID'].apply(lambda x: int(x.split('of:')[1], 16)) # Convert from hex to decimal
        self.data['Port Number'] = self.data['Port Number'].apply(lambda x: int(x.split('Port#:')[1])) # Convert 'Port Number' to integer

    def preprocess_data(self):
        label_encoder = LabelEncoder()
        self.data['Label'] = label_encoder.fit_transform(self.data['Label'])

    def split_data(self, test_size=0.2, random_state=42):
        X = self.data.drop(['Label', 'Binary Label'], axis=1)
        y = self.data['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def scale_data(self, X_train, X_test):
        scaler = StandardScaler()
        numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
        X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])
        return X_train_scaled, X_test_scaled

class ModelTrainer:
    def __init__(self, model):
        self.model = model

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

class Evaluator:
    @staticmethod
    def evaluate_accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def evaluate_classification_report(y_true, y_pred):
        class_labels = np.unique(np.concatenate((y_true, y_pred)))
        class_labels_str = [str(label) for label in class_labels] # added this conversion line here
        return classification_report(y_true, y_pred, target_names=class_labels_str)

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred):
        conf_matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

def run_mainrfr():
    file_path = r'UNR-IDD.csv'

    data_processor = DataProcessor(file_path)
    data_processor.feature_engineering()
    data_processor.preprocess_data()

    X_train, X_test, y_train, y_test = data_processor.split_data()

    X_train_scaled, X_test_scaled = data_processor.scale_data(X_train, X_test)

    rf_classifier = RandomForestClassifier(random_state=42)
    model_trainer = ModelTrainer(rf_classifier)
    model_trainer.train_model(X_train_scaled, y_train)

    y_pred = model_trainer.predict(X_test_scaled)

    accuracy = Evaluator.evaluate_accuracy(y_test, y_pred)
    print("Random Forest Test Accuracy:", accuracy)

    class_report = Evaluator.evaluate_classification_report(y_test, y_pred)
    print("Classification Report:")
    print(class_report)

    Evaluator.plot_confusion_matrix(y_test, y_pred)

# Main script
if __name__ == "__main__":
    run_mainrfr()
