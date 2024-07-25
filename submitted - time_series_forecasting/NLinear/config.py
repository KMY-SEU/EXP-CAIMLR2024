import argparse

parser = argparse.ArgumentParser()

# data path
parser.add_argument('--data_file', default='ETTm2.csv')
parser.add_argument('--save_path', default='./saved_model/')

# preprocess data
parser.add_argument('--seq_len', default=40, help='len_hist')
parser.add_argument('--pred_len', default=40, help='len_pred')
parser.add_argument('--do_normalization', default=True)

# devices
parser.add_argument('--use_cuda', default=False)
parser.add_argument('--use_parallel', default=False)

# modeling
parser.add_argument('--individual', default=False)
parser.add_argument('--enc_in', default=7, help='depth_input')
# parser.add_argument('--hidden_size', default=128)


# training
parser.add_argument('--learning_rate', default=0.0001)
parser.add_argument('--training_epoch', default=10000)
parser.add_argument('--batch_size', default=64)
parser.add_argument('--early_stop', default=5)

# parse arguments
args = parser.parse_args()
