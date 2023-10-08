#!/usr/bin/env python3

__author__ = 'Dmitry Ustalov'
__license__ = 'GPL-3.0-or-later'

import argparse
import random
import typing
from collections import defaultdict, Counter
from functools import partial
from operator import itemgetter

import networkx as nx
import pandas as pd
from Levenshtein import distance
from tqdm.auto import tqdm


def graph(df_slice: pd.DataFrame, n: int = 3) -> nx.Graph:
    counts: typing.DefaultDict[str, typing.Counter[str]] = defaultdict(Counter)

    for _, a in df_slice.iterrows():
        for _, b in df_slice.iterrows():
            if a['model'] != b['model']:
                counts[a['model']][b['model']] += distance(
                    a['result'].strip().lower(),
                    b['result'].strip().lower()
                )

    G = nx.Graph()

    G.add_nodes_from(df_slice['model'])

    for _, row in df_slice.iterrows():
        for top, weight in counts[row['model']].most_common(n):
            G.add_edge(row['model'], top, weight=weight)

    assert nx.is_connected(G), 'G should be connected'

    return G


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', type=argparse.FileType('rb'), default='prompts.jsonl')
    parser.add_argument('--models', type=argparse.FileType('rb'), default='models.jsonl')
    parser.add_argument('--results', type=argparse.FileType('rb'), default='results.jsonl')
    parser.add_argument('--pairs', type=argparse.FileType('wb'), default='pairs.jsonl')
    parser.add_argument('--pairs-basic', type=argparse.FileType('wb'), default='pairs-basic.jsonl')
    parser.add_argument('-n', '--neighbors', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    df_prompts = pd.read_json(args.prompts, lines=True)
    df_prompts.set_index('id', inplace=True)

    df_models = pd.read_json(args.models, lines=True)
    df_models.set_index('id', inplace=True)

    df_results = pd.read_json(args.results, lines=True)
    df_results.set_index('id', inplace=True)

    tqdm.pandas(desc='Graphs')
    df_prompts['graph'] = df_results.groupby('prompt').progress_apply(partial(graph, n=args.neighbors))
    df_prompts['pairs'] = df_prompts['graph'].apply(lambda G: G.edges)

    df_pairs = df_prompts.explode('pairs')
    df_pairs.reset_index(inplace=True)
    del df_pairs['stop'], df_pairs['graph']

    df_pairs['lmodel'] = df_pairs['pairs'].apply(itemgetter(0))
    df_pairs['rmodel'] = df_pairs['pairs'].apply(itemgetter(1))

    del df_pairs['pairs']

    for model_column in ('lmodel', 'rmodel'):
        df_pairs = pd.merge(df_pairs, df_results[['model', 'prompt', 'result']],
                            left_on=('id', model_column), right_on=('prompt', 'model'))

    assert (df_pairs['lmodel'] == df_pairs['model_x']).all()
    assert (df_pairs['rmodel'] == df_pairs['model_y']).all()
    assert (df_pairs['id'] == df_pairs['prompt_x']).all()
    assert (df_pairs['id'] == df_pairs['prompt_y']).all()

    del df_pairs['lmodel'], df_pairs['rmodel']
    del df_pairs['prompt_x'], df_pairs['prompt_y']

    rng = random.Random(args.seed)

    df_pairs['swap'] = [rng.choice([True, False]) for _ in range(len(df_pairs))]

    for prefix in ('model', 'result'):
        df_pairs.loc[df_pairs['swap'], f'{prefix}_x'], df_pairs.loc[df_pairs['swap'], f'{prefix}_y'] = \
            df_pairs.loc[df_pairs['swap'], f'{prefix}_y'], df_pairs.loc[df_pairs['swap'], f'{prefix}_x']

    del df_pairs['swap']

    df_pairs.rename(columns={
        'id': 'prompt'
    }, inplace=True)

    df_pairs.index.name = 'id'

    df_pairs.reset_index(inplace=True)
    df_pairs.to_json(args.pairs, orient='records', lines=True)

    df_pairs = df_pairs[(df_pairs['type'] != 'code') & ~(df_pairs['slug'].isin({'cot-sally', 'svg'}))]
    df_pairs.to_json(args.pairs_basic, orient='records', lines=True)


if __name__ == '__main__':
    main()
