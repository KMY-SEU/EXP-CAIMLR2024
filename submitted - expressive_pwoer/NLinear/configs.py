import argparse

parser = argparse.ArgumentParser()

# preprocess data
parser.add_argument('--seq_len', default=40, help='len_hist')
parser.add_argument('--pred_len', default=40, help='len_pred')
parser.add_argument('--do_normalization', default=True)

# modeling
parser.add_argument('--individual', default=False)
parser.add_argument('--enc_in', default=2, help='depth_input')

# parse arguments
args = parser.parse_args()
