import argparse
import os
import re
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('root', type=str)
parser.add_argument(
        '--prefix', type=str, default='', help='Prefix for output names')
args = parser.parse_args()


for task, target_metric in [('segmentation', 'dice'),
                            ('classification', 'f1'),
                            ('detection', 'f1')]:

    pat = f'{args.prefix}(?P<fold>.+)_{task}_scores.csv'

    scores = []
    for filename in os.listdir(args.root):
        if re.match(pat, filename):
            fold = re.match(pat, filename).group('fold')
            score = pd.read_csv(os.path.join(args.root, filename), index_col=0)
            scores.append(score)


    if len(scores) > 0:
        df = pd.DataFrame(columns=scores[0].columns, index=scores[0].index)
        scores = [score.reindex(df.index) for score in scores]

        for column in df.columns:
            values = [score[column].values for score in scores]
            values = np.array(values)

            means = values.mean(axis=0)
            stds = values.std(axis=0)

            df[column] = [f'{mean:.3f} +- {std:.3f}' for mean, std in zip(means, stds)]

        df.insert(0, '# folds', len(scores))

        df['target'] = df[target_metric].apply(lambda x: float(x.split('+-')[0].strip()))
        df = df.sort_values(by=['target'], ascending=False)
        df = df.drop(['target'], axis=1)

        print(df)
        df.to_csv(os.path.join(args.root, args.prefix + f'{task}_scores.csv'))