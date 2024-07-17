import os
import re
import tokenize
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from skorch import NeuralNetClassifier

# Global list to store file paths with tokenization errors
error_files = []

# Preprocessing functions
def tokenize_code(code, file_path):
    tokens = []
    reader = BytesIO(code.encode('utf-8')).readline
    try:
        for toknum, tokval, _, _, _ in tokenize.tokenize(reader):
            if toknum != tokenize.ENCODING:
                tokens.append(tokval)
    except tokenize.TokenError as e:
        error_files.append(file_path)
        print("Error tokenizing code in file:", file_path)
    return tokens

def normalize_code(code):
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'\s+', ' ', code).strip()
    return code

def preprocess_code(code, file_path):
    normalized_code = normalize_code(code)
    tokens = tokenize_code(normalized_code, file_path)
    return ' '.join(tokens)

# Load data from directories

def load_data_from_directory(directory, label):
    data = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".sol"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                try:
                    code = file.read()
                    preprocessed_code = preprocess_code(code, filepath)
                    data.append(preprocessed_code)
                    labels.append(label)
                except Exception as e:
                    error_files.append(filepath)
                    print("Error processing file:", filepath)
    return data, labels

# Paths to the directories
vulnerable_dir = './Code/Contracts for training/backup/Re-entrancy'
non_vulnerable_dir = './Code/Contracts for training/backup/Verified'

# Load and label the data
vulnerable_data, vulnerable_labels = load_data_from_directory(vulnerable_dir, 1)
non_vulnerable_data, non_vulnerable_labels = load_data_from_directory(non_vulnerable_dir, 0)

# Combine the data and labels
data = vulnerable_data + non_vulnerable_data
labels = vulnerable_labels + non_vulnerable_labels

# Vectorize the data using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data).toarray().astype('float32')  # Ensure dtype float32
y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # Change dtype to float32 and reshape to [batch_size, 1]

# Define the neural network model
class SmartContractVulnerabilityModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64):
        super(SmartContractVulnerabilityModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Output should not apply sigmoid here for BCEWithLogitsLoss
        return x

# Skorch wrapper for the PyTorch model
net = NeuralNetClassifier(
    SmartContractVulnerabilityModel,
    module__input_dim=1000,  # This should match the input dimension used during training
    max_epochs=10,
    lr=0.001,
    optimizer=optim.Adam,
    criterion=nn.BCEWithLogitsLoss,  # Use BCEWithLogitsLoss which combines sigmoid and BCE in a more numerically stable way
    iterator_train__shuffle=True,
)

# Hyperparameter grid
params = {
    'lr': [0.001, 0.01, 0.1],
    'max_epochs': [10, 20],
    'module__hidden_dim1': [128, 256],
    'module__hidden_dim2': [64, 128],
}

# Initialize GridSearchCV
gs = GridSearchCV(net, params, refit=True, cv=5, scoring='accuracy')

# Perform grid search
gs.fit(X, y)

# Print the best parameters and the best score
print("Best parameters found:", gs.best_params_)
print("Best score:", gs.best_score_)

# Save the best model
torch.save(gs.best_estimator_.module_.state_dict(), 'smart_contract_vulnerability_model_best.pth')

# Write the error log to a text file
with open('error_log.txt', 'w', encoding='utf-8') as error_file:
    for filepath in error_files:
        error_file.write(filepath + '\n')
