'''
    Author: kangmingyu
    Email: kangmingyu@seu.edu.cn
    Institute: CCCS Lab, Southeast University
'''

import argparse

parser = argparse.ArgumentParser()

# preprocess data
parser.add_argument('--seq_len', type=int, default=40, help='input sequence length')
parser.add_argument('--enc_in', default=2)
parser.add_argument('--d_model', default=128)
parser.add_argument('--n_layers', default=2)

# parse arguments
args = parser.parse_args()
