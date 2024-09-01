from solcx import compile_standard

with open("../Contracts for training/reentrant_contracts/0x0da76de0916ef2da3c58a97e4d09d501c56a9f15.sol", "r") as file:
    solidity_code = file.read()

compiled_sol = compile_standard({
    "language": "Solidity",
    "sources": {
        "contract.sol": {
            "content": solidity_code
        }
    },
    "settings": {
        "outputSelection": {
            "*": {
                "*": ["ast"]
            }
        }
    }
})

# Extract the AST
ast = compiled_sol['contracts']['contract.sol']['ast']

# Example: Traverse the AST to gather features
def traverse(node, features):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == 'name':
                features.append(f"node_{value}")
            traverse(value, features)
    elif isinstance(node, list):
        for item in node:
            traverse(item, features)

features = []
traverse(ast, features)

# Now, you can use these features to train your ML model
print(features)
