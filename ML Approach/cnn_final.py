import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import numpy as np

class DataPreprocessor:
    def __init__(self, train_path, val_path, test_path):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

    def load_data(self):
        train = pd.read_csv(self.train_path)
        val = pd.read_csv(self.val_path)
        test = pd.read_csv(self.test_path)
        return train, val, test

    def preprocess_data(self, train, val, test):
        # Encoding categorical variables
        train = pd.get_dummies(train, columns=['Switch ID', 'Port Number'])
        val = pd.get_dummies(val, columns=['Switch ID', 'Port Number'])
        test = pd.get_dummies(test, columns=['Switch ID', 'Port Number'])

        # Define features and target
        X = train.drop('Label', axis=1)
        y = train['Label']

        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X, y)

        # Split the data into features and target for test and val
        X_test = test.drop('Label', axis=1)
        y_test = test['Label']
        X_val = val.drop('Label', axis=1)
        y_val = val['Label']

        # Encode target labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_val_encoded = label_encoder.transform(y_val)
        y_test_encoded = label_encoder.transform(y_test)

        # Scale numeric columns
        scaler = StandardScaler()
        numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
        X_val_scaled[numeric_columns] = scaler.transform(X_val[numeric_columns])
        X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])

        # Handle infinities and fill NaNs
        X_train_scaled = X_train_scaled.astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0)
        X_val_scaled = X_val_scaled.astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test_scaled = X_test_scaled.astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0)

        # Convert to tensors
        train_features = torch.tensor(X_train_scaled.values.astype(np.float32))
        val_features = torch.tensor(X_val_scaled.values.astype(np.float32))
        test_features = torch.tensor(X_test_scaled.values.astype(np.float32))

        train_labels = torch.tensor(y_train_encoded.astype(np.int64))
        val_labels = torch.tensor(y_val_encoded.astype(np.int64))
        test_labels = torch.tensor(y_test_encoded.astype(np.int64))

        # Create TensorDatasets and DataLoaders
        train_dataset = TensorDataset(train_features, train_labels)
        val_dataset = TensorDataset(val_features, val_labels)
        test_dataset = TensorDataset(test_features, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return train_loader, val_loader, test_loader, len(np.unique(y_train_encoded))

class ModelTrainer:
    def __init__(self, input_dim, num_classes, device):
        self.device = device
        self.model = CNN(input_dim, num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def train_model(self, train_loader, epochs=100):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for data, targets in train_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

    def evaluate_model(self, data_loader):
        self.model.eval()
        actuals = []
        predictions = []

        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.view(-1).cpu().numpy())
                actuals.extend(targets.view(-1).cpu().numpy())

        mse = mean_squared_error(actuals, predictions)
        f1 = f1_score(actuals, predictions, average=None)
        precision = precision_score(actuals, predictions, average='macro')
        recall = recall_score(actuals, predictions, average='macro')
        accuracy = accuracy_score(actuals, predictions)

        print(f'MSE: {mse:.4f}')
        for i, score in enumerate(f1):
            print(f'F1 Score for class {i}: {score:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'Accuracy: {accuracy:.4f}')

class CNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(CNN, self).__init__()
        self.layer1 = nn.Conv1d(1, 16, 3, padding=1)  # Using padding to preserve dimensions
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(16 * (num_features // 2), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = self.pool(self.relu(self.layer1(x)))
        x = x.view(-1, 16 * (x.size(2)))  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Example usage
def run_maincnn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_processor = DataPreprocessor('/Users/admin/Downloads/train.csv', '/Users/admin/Downloads/val.csv', '/Users/admin/Downloads/test.csv')
    train_loader, val_loader, test_loader, num_classes = data_processor.preprocess_data(*data_processor.load_data())

    model_trainer = ModelTrainer(train_loader.dataset.tensors[0].shape[1], num_classes, device)
    model_trainer.train_model(train_loader)
    model_trainer.evaluate_model(test_loader)


if __name__ == '__main__':
    run_maincnn()
