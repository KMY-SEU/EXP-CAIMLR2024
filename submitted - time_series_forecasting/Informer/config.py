import argparse

parser = argparse.ArgumentParser()

# data path
parser.add_argument('--data_file', default='ETTm2.csv')
parser.add_argument('--save_path', default='./saved_model/')

# preprocess data
parser.add_argument('--do_normalization', default=True)

# devices
parser.add_argument('--use_cuda', default=False)
parser.add_argument('--use_parallel', default=False)

# modeling
parser.add_argument('--seq_len', type=int, default=12, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=12, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

# parser.add_argument('--itr', type=int, default=2, help='experiments times')
# parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
# parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
# parser.add_argument('--des', type=str, default='test',help='exp description')
# parser.add_argument('--loss', type=str, default='mse',help='loss function')
# parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
# parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
# parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)


# training
parser.add_argument('--learning_rate', default=0.0001)
parser.add_argument('--training_epoch', default=10000)
parser.add_argument('--batch_size', default=64)
parser.add_argument('--early_stop', default=5)

# parse arguments
args = parser.parse_args()
