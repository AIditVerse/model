import torch
import torch.nn as nn
import re
import tokenize
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Define the model architecture
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

# Load the trained model
input_dim = 1000
model = SmartContractVulnerabilityModel(input_dim=input_dim, hidden_dim1=256, hidden_dim2=128)
model.load_state_dict(torch.load('AI Model/Model Training/neural_network_model.pth'))
model.eval()

# Load the saved vectorizer
vectorizer = joblib.load('AI Model/Model Training/neural_network_vectors.pkl')

# Preprocessing functions
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

# Preprocess new code
def preprocess_new_code(code):
    normalized_code = normalize_code(code)
    tokens = tokenize_code(normalized_code, "new_code.sol")
    return ' '.join(tokens)

# Example new code
new_code = """

"""

# Preprocess and vectorize the new code
preprocessed_new_code = preprocess_new_code(new_code)
X_new = vectorizer.transform([preprocessed_new_code]).toarray().astype('float32')

# Predict using the trained model
with torch.no_grad():
    y_new_pred = model(torch.tensor(X_new))
    prediction = torch.sigmoid(y_new_pred).round().item()

print("Prediction for the new code:", "Vulnerable" if prediction == 1.0 else "Not Vulnerable")
