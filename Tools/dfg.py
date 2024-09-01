from slither import Slither

# Initialize Slither with the Solidity file
slither = Slither('./Contracts for training/reentrant_contracts/0x0da76de0916ef2da3c58a97e4d09d501c56a9f15.sol')

# Extract data flow graph
dfg = slither.get_dataflow()

# Example: Traverse the DFG to gather features
features = []
for node in dfg:
    for variable in node.variables_read:
        features.append(f"read_{variable.name}")
    for variable in node.variables_written:
        features.append(f"written_{variable.name}")

# Now, you can use these features to train your ML model
print(features)
