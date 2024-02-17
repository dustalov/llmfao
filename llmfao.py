#!/usr/bin/env python3

__author__ = 'Dmitry Ustalov'
__license__ = 'GPL-3.0-or-later'

import argparse
import re
from collections import defaultdict, Counter
from contextlib import nullcontext
from functools import partial
from operator import itemgetter
from random import Random
from string import Template
from typing import Any

import networkx as nx
import pandas as pd
from Levenshtein import distance
from tqdm.auto import tqdm


def graph(df_slice: pd.DataFrame, n: int = 3) -> 'nx.Graph[str]':
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for model_a, result_a in df_slice[['model', 'result']].itertuples(index=False):
        for model_b, result_b in df_slice[['model', 'result']].itertuples(index=False):
            if model_a != model_b:
                counts[model_a][model_b] += distance(
                    result_a.strip().lower(),
                    result_b.strip().lower()
                )

    G: nx.Graph[str] = nx.Graph()

    G.add_nodes_from(df_slice['model'])

    for model in df_slice['model']:
        for top, weight in counts[model].most_common(n):
            G.add_edge(model, top, weight=weight)

    assert nx.is_connected(G), 'G should be connected'

    return G


def pairs(args: argparse.Namespace) -> None:
    df_prompts = pd.read_json(args.prompts, lines=True)
    df_prompts.set_index('id', inplace=True)

    df_models = pd.read_json(args.models, lines=True)
    df_models.set_index('id', inplace=True)

    df_results = pd.read_json(args.results, lines=True)
    df_results.set_index('id', inplace=True)

    tqdm.pandas(desc='Graphs')
    df_prompts['graph'] = df_results.groupby('prompt').progress_apply(partial(graph, n=args.neighbors))  # type: ignore
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

    rng = Random(args.seed)

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
    df_pairs.to_json(args.pairs_crowd, orient='records', lines=True)


def build_gpt_request_metadata(row: 'pd.Series[Any]') -> dict[str, int]:
    return {
        'id': row['id'],
        'prompt': row['prompt'],
        'model_x': row['model_x'],
        'model_y': row['model_y']
    }


def gpt3_requests(args: argparse.Namespace) -> None:
    df = pd.read_json(args.pairs, lines=True)

    template = Template(args.instruction.read())

    df['metadata'] = df.apply(build_gpt_request_metadata, axis=1)
    df['max_tokens'] = 3
    df['model'] = 'gpt-3.5-turbo-instruct'
    df['prompt'] = df.apply(lambda row: template.safe_substitute(
        text=row['text'].strip(),
        result_x=row['result_x'].strip(),
        result_y=row['result_y'].strip()
    ).strip(), axis=1)

    df = df[['model', 'prompt', 'metadata', 'max_tokens']]

    df.to_json(args.output, orient='records', lines=True)


def gpt4_requests(args: argparse.Namespace) -> None:
    df = pd.read_json(args.pairs, lines=True)

    template = Template(args.instruction.read())

    df['metadata'] = df.apply(build_gpt_request_metadata, axis=1)
    df['max_tokens'] = 3
    df['model'] = 'gpt-4'

    def chat_completion(row: 'pd.Series[Any]') -> list[dict[str, str]]:
        prompt = template.safe_substitute(
            text=row['text'].strip(),
            result_x=row['result_x'].strip(),
            result_y=row['result_y'].strip()
        ).strip()

        system, _, user = prompt.partition("\n")

        return [
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user.strip()}
        ]

    df['messages'] = df.apply(chat_completion, axis=1)
    df = df[['model', 'messages', 'metadata', 'max_tokens']]

    df.to_json(args.output, orient='records', lines=True)


def join_gpt_responses(df: pd.DataFrame, df_pairs: pd.DataFrame, df_models: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat((pd.json_normalize(df['metadata']), df), axis=1)  # type: ignore
    df.set_index('id', inplace=True)
    del df['input'], df['completion'], df['metadata']

    df = pd.merge(df, df_models[['name']], left_on='model_x', right_index=True)
    df = pd.merge(df, df_models[['name']], left_on='model_y', right_index=True)

    df.drop_duplicates(['prompt', 'model_x', 'model_y'], keep='last', inplace=True)

    df = pd.merge(df_pairs[['id', 'prompt', 'model_x', 'model_y']], df, on=['prompt', 'model_x', 'model_y'])

    df.rename(columns={
        'name_x': 'left',
        'name_y': 'right'
    }, inplace=True)

    df.set_index('id', inplace=True)

    assert len(df) == len(df_pairs), f'Lengths mismatch: {len(df_pairs)} != {len(df)}'

    return df


def gpt3_comparisons(args: argparse.Namespace) -> None:
    df_pairs = pd.read_json(args.pairs, lines=True)

    df_models = pd.read_json(args.models, lines=True)
    df_models.set_index('id', inplace=True)

    df = pd.read_json(args.responses, lines=True)
    df.columns = pd.Index(['input', 'completion', 'metadata'])
    df['winner'] = df['completion'].apply(lambda completion: completion['choices'][0]['text'].strip())

    df = join_gpt_responses(df, df_pairs, df_models)

    df.loc[df['winner'].str.contains(r'neither', flags=re.IGNORECASE), 'winner'] = 'tie'
    df.loc[df['winner'].str.contains(r'(?:^|\b)A(?:\b|$)'), 'winner'] = 'left'
    df.loc[df['winner'].str.contains(r'(?:^|\b)B(?:\b|$)'), 'winner'] = 'right'
    df.loc[~df['winner'].isin({'left', 'right'}), 'winner'] = 'tie'

    assert all(winner in {'left', 'right', 'tie'} for winner in df['winner'].unique()), 'odd GPT-3 winners'

    df = df[df['prompt'].isin(df_pairs['prompt'])]
    df.to_csv(args.output, sep=args.delimiter)


def gpt4_comparisons(args: argparse.Namespace) -> None:
    df_pairs = pd.read_json(args.pairs, lines=True)

    df_models = pd.read_json(args.models, lines=True)
    df_models.set_index('id', inplace=True)

    df = pd.read_json(args.responses, lines=True)
    df.columns = pd.Index(['input', 'completion', 'metadata'])

    df['success'] = df['completion'].apply(
        lambda completions: isinstance(completions, dict) and 'choices' in completions
    )

    df.loc[df['success'], 'winner'] = df.loc[df['success'], 'completion'].apply(
        lambda completion: completion['choices'][0]['message']['content'].strip()
    )

    del df['success']

    df = join_gpt_responses(df, df_pairs, df_models)

    df.fillna({'winner': 'tie'}, inplace=True)
    df.loc[df['winner'].str.lower().str.contains('neither'), 'winner'] = 'tie'
    df.loc[df['winner'].str.contains(r'(?:^|\b)A(?:\b|$)'), 'winner'] = 'left'
    df.loc[df['winner'].str.contains(r'(?:^|\b)B(?:\b|$)'), 'winner'] = 'right'
    df.loc[~df['winner'].isin({'left', 'right'}), 'winner'] = 'tie'

    assert all(winner in {'left', 'right', 'tie'} for winner in df['winner'].unique()), 'odd GPT-4 winners'

    df = df[df['prompt'].isin(df_pairs['prompt'])]
    df.to_csv(args.output, sep=args.delimiter)


def main() -> None:
    parser = argparse.ArgumentParser(description='LLMFAO: Large Language Model Feedback Analysis and Optimization')

    subparsers = parser.add_subparsers()

    with nullcontext(subparsers.add_parser('pairs')) as subparser:
        subparser.add_argument('--prompts', type=argparse.FileType('rb'), default='prompts.jsonl')
        subparser.add_argument('--models', type=argparse.FileType('rb'), default='models.jsonl')
        subparser.add_argument('--results', type=argparse.FileType('rb'), default='results.jsonl')
        subparser.add_argument('--pairs', type=argparse.FileType('wb'), default='pairs.jsonl')
        subparser.add_argument('--pairs-crowd', type=argparse.FileType('wb'), default='pairs-crowd.jsonl')
        subparser.add_argument('-n', '--neighbors', type=int, default=3)
        subparser.add_argument('--seed', type=int, default=0)
        subparser.set_defaults(func=pairs)

    with nullcontext(subparsers.add_parser('gpt3-requests')) as subparser:
        subparser.add_argument('--pairs', type=argparse.FileType('rb'), default='pairs.jsonl')
        subparser.add_argument('--instruction', type=argparse.FileType('r', encoding='UTF-8'),
                               default='gpt-instruction.txt')
        subparser.add_argument('--output', type=argparse.FileType('wb'), default='gpt3.jsonl')
        subparser.set_defaults(func=gpt3_requests)

    with nullcontext(subparsers.add_parser('gpt4-requests')) as subparser:
        subparser.add_argument('--pairs', type=argparse.FileType('rb'), default='pairs.jsonl')
        subparser.add_argument('--instruction', type=argparse.FileType('r', encoding='UTF-8'),
                               default='gpt-instruction.txt')
        subparser.add_argument('--output', type=argparse.FileType('wb'), default='gpt4.jsonl')
        subparser.set_defaults(func=gpt4_requests)

    with nullcontext(subparsers.add_parser('gpt3-comparisons')) as subparser:
        subparser.add_argument('--pairs', type=argparse.FileType('rb'), default='pairs.jsonl')
        subparser.add_argument('--models', type=argparse.FileType('rb'), default='models.jsonl')
        subparser.add_argument('--responses', type=argparse.FileType('rb'), default='gpt3-responses.jsonl')
        subparser.add_argument('-d', '--delimiter', type=str, default=',')
        subparser.add_argument('--output', type=argparse.FileType('wb'), default='gpt3-comparisons.csv')
        subparser.set_defaults(func=gpt3_comparisons)

    with nullcontext(subparsers.add_parser('gpt4-comparisons')) as subparser:
        subparser.add_argument('--pairs', type=argparse.FileType('rb'), default='pairs.jsonl')
        subparser.add_argument('--models', type=argparse.FileType('rb'), default='models.jsonl')
        subparser.add_argument('--responses', type=argparse.FileType('rb'), default='gpt4-responses.jsonl')
        subparser.add_argument('-d', '--delimiter', type=str, default=',')
        subparser.add_argument('--output', type=argparse.FileType('wb'), default='gpt4-comparisons.csv')
        subparser.set_defaults(func=gpt4_comparisons)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
