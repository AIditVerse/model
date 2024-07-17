import os
import re
import tokenize
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from flask import Flask, request, jsonify
import joblib

# Define the preprocessing functions
def tokenize_code(code, file_path):
    tokens = []
    reader = BytesIO(code.encode('utf-8')).readline
    try:
        for toknum, tokval, _, _, _ in tokenize.tokenize(reader):
            if toknum != tokenize.ENCODING:
                tokens.append(tokval)
    except tokenize.TokenError as e:
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
                    print("Error processing file:", filepath)
    return data, labels

# Load and label the data
vulnerable_dir = './Contracts for training/Re-entrancy'
non_vulnerable_dir = './Contracts for training/Verified'

vulnerable_data, vulnerable_labels = load_data_from_directory(vulnerable_dir, 1)
non_vulnerable_data, non_vulnerable_labels = load_data_from_directory(non_vulnerable_dir, 0)

data = vulnerable_data + non_vulnerable_data
labels = vulnerable_labels + non_vulnerable_labels

# Vectorize the data using TF-IDF
vectorizer = joblib.load('neural_network_vectors.pkl')
X = vectorizer.fit_transform(data).toarray().astype('float32')
y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

# Define the neural network model
class SmartContractVulnerabilityModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1=256, hidden_dim2=128):
        super(SmartContractVulnerabilityModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the pre-trained model
model = SmartContractVulnerabilityModel(input_dim=1000, hidden_dim1=256, hidden_dim2=128)
model.load_state_dict(torch.load('smart_contract_vulnerability_model_best.pth'))
model.eval()

# Skorch wrapper for the PyTorch model
net = NeuralNetClassifier(
    module=model,
    module__input_dim=1000,
    max_epochs=20,
    lr=0.001,
    optimizer=optim.Adam,
    criterion=nn.BCEWithLogitsLoss,
    iterator_train__shuffle=True,
    callbacks=[EarlyStopping(patience=5)],
)

# Function to analyze a new code file
def analyze_code_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        code = file.read()
    preprocessed_code = preprocess_code(code, file_path)
    X_new = vectorizer.transform([preprocessed_code]).toarray().astype('float32')
    y_new_pred = net.predict(X_new)
    return y_new_pred

# Example usage
file_path = 'path_to_your_file.sol'
prediction = analyze_code_file(file_path)
print("Prediction for the new code:", prediction)

# Flask server setup for the backend process
app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    
    prediction = analyze_code_file(file_path)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
