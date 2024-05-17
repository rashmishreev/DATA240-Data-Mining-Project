# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score, classification_report

class DataProcessor:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.label_encoder = LabelEncoder()

    def perform_eda(self):
        missing_values = self.data.isnull().sum()
        print("Missing values:\n", missing_values)

        class_distribution = self.data['Binary Label'].value_counts(normalize=True)
        print("\nClass Distribution:\n", class_distribution)

        plt.figure(figsize=(8, 6))
        sns.countplot(x='Binary Label', data=self.data)
        plt.title("Class Distribution")
        plt.show()

    def encode_labels(self):
        self.data['Binary Label'] = self.label_encoder.fit_transform(self.data['Binary Label'])

    def preprocess_data(self):
        X = self.data.drop(columns=['Switch ID', 'Port Number', 'Label', 'Binary Label'])
        y = self.data['Binary Label']

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        selector = SelectKBest(score_func=chi2, k=10)
        X_selected = selector.fit_transform(X_scaled, y)

        return train_test_split(X_selected, y, test_size=0.6, random_state=42)

class KNNModel:
    def __init__(self, param_grid=None):
        self.param_grid = param_grid if param_grid else {'n_neighbors': range(3, 12), 'weights': ['uniform', 'distance']}
        self.best_model = None

    def perform_grid_search(self, X_train, y_train):
        grid_search = GridSearchCV(KNeighborsClassifier(), self.param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        self.best_model = grid_search.best_estimator_
        print("\nBest KNN parameters:", grid_search.best_params_)

    def evaluate_model(self, X_train, y_train):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.best_model, X_train, y_train, cv=cv)
        print("KNN Cross-Validation Accuracy:", cv_scores.mean())

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        self.best_model.fit(X_train, y_train)
        y_pred = self.best_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print("\nModel Evaluation for Binary Classification:")
        print("MSE:", mse)
        print("F1 Score:", f1)
        print("Precision:", precision)
        print("Recall:", recall)

        print("\nKNN Classification Report (Binary):")
        print(classification_report(y_test, y_pred))

class Visualizer:
    @staticmethod
    def plot_pairplot(data):
        sampled_data = data.sample(n=5000, random_state=42)
        sns.pairplot(sampled_data, hue='Binary Label', palette='Set1')
        plt.suptitle("Pairplot of Features", y=1.02)
        plt.show()

    @staticmethod
    def plot_histogram(data):
        plt.figure(figsize=(8, 6))
        sns.histplot(data['Received Bytes'], bins=30, kde=True)
        plt.title("Distribution of Received Bytes")
        plt.xlabel("Received Bytes")
        plt.ylabel("Frequency")
        plt.show()

    @staticmethod
    def plot_correlation_heatmap(data, relevant_cols):
        numeric_data = data[relevant_cols]
        corr_matrix = numeric_data.corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()

def run_mainknn():
    file_path = "UNR-IDD.csv"

    data_processor = DataProcessor(file_path)
    data_processor.perform_eda()
    data_processor.encode_labels()

    X_train, X_test, y_train, y_test = data_processor.preprocess_data()

    # Adding noise to the test data
    np.random.seed(42)
    noise = np.random.normal(0, 0.4, X_test.shape)
    X_test += noise

    knn_model = KNNModel()
    knn_model.perform_grid_search(X_train, y_train)
    knn_model.evaluate_model(X_train, y_train)
    knn_model.train_and_evaluate(X_train, X_test, y_train, y_test)

    # Visualizations
    relevant_cols = ['Received Packets', 'Received Bytes', 'Sent Bytes', 'Sent Packets', 'Port alive Duration (S)', 'Packets Rx Dropped',
                     'Packets Tx Dropped', 'Packets Rx Errors', 'Packets Tx Errors', 'Total Load/Rate', 'Total Load/Latest', 'Unknown Load/Rate',
                     'Unknown Load/Latest', 'Latest bytes counter', 'Active Flow Entries', 'Packets Looked Up', 'Packets Matched']

    Visualizer.plot_pairplot(data_processor.data)
    Visualizer.plot_histogram(data_processor.data)
    Visualizer.plot_correlation_heatmap(data_processor.data, relevant_cols)

if __name__ == "__main__":
    run_mainknn()
