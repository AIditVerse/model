import sys
import pprint

from solidity_parser import parser

# Provide the file path here
file_path = './smart-contracts/dataset/CV/hardcoded_gas_limits.sol'  # Adjust the path as per your file location

sourceUnit = parser.parse_file(file_path, loc=True)  # loc=True -> add location information to ast nodes
pprint.pprint(sourceUnit)

# see output below

# python <file name> <solidity code file name>
# python '.\sol to ast parser.py' .\ERC6551Registry.sol 
# put the above command in terminal and the code will run

