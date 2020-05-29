import argparse
import os
import re
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('root', type=str)
parser.add_argument(
        '--prefix', type=str, default='', help='Prefix for output names')
args = parser.parse_args()

df = None
for filename in os.listdir(args.root):
    pat = args.prefix + r'(?P<fold>.+)_scores.csv'
    if re.match(pat, filename):
        fold = re.match(pat, filename).group('fold')
        scores = pd.read_csv(os.path.join(args.root, filename), index_col=0)
        if df is None:
            df = pd.DataFrame(columns=scores.columns)
            df.index.name = 'Fold'
        df.loc[fold] = scores.loc['MEAN']

df = df.sort_index()

df.loc["Mean"] = df.mean(axis=0)
df.loc["Std"] = df.std(axis=0)

df.loc['CV'] = [f'{mean:.3f} +- {std:.3f}' for mean, std in zip(df.loc["Mean"], df.loc["Std"] )]

print(df)
df.to_csv(os.path.join(args.root, args.prefix + 'scores.csv'))