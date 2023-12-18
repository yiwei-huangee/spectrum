import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--run_mode', type=str, default='train')
parser.add_argument('--using_arival_time', type=str, default='real')

parser.add_argument('--N', type=int, default=2000)
parser.add_argument('--input_size', type=int, default=1024)
parser.add_argument('--dim_attn', type=int, default=64)
parser.add_argument('--dim_val', type=int, default=64)
parser.add_argument('--n_heads', type=int, default=4)
args = parser.parse_args()