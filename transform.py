#!/usr/bin/env python3

__author__ = 'Dmitry Ustalov'
__license__ = 'GPL-3.0-or-later'

import argparse
import re
from contextlib import nullcontext
from functools import partial
from random import Random
from string import Template
from typing import Dict, List, cast

import pandas as pd


def build_gpt_request_metadata(row: pd.Series) -> Dict[str, int]:
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

    def chat_completion(row: pd.Series) -> List[Dict[str, str]]:
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
    df = pd.concat((pd.json_normalize(df['metadata']), df), axis=1)
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
    df.columns = ['input', 'completion', 'metadata']
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
    df.columns = ['input', 'completion', 'metadata']

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


def aggregated(args: argparse.Namespace) -> None:
    def aggregate(df_slice: pd.DataFrame, rng: Random) -> str:
        counts = df_slice['winner'].value_counts()

        assert not counts.empty, 'counts is empty'

        max_count = counts.max()
        modes = counts[counts == max_count].index

        return cast(str, rng.choice(modes))

    dfs: List[pd.DataFrame] = []

    for file in args.inputs:
        df_input = pd.read_csv(file, sep=args.delimiter)

        rng = Random(args.seed)

        df_aggregated = df_input.groupby(['id', 'prompt', 'model_x', 'model_y', 'left', 'right']).apply(
            partial(aggregate, rng=rng)
        ).reset_index(name='winner')

        dfs.append(df_aggregated)

    df = pd.concat(dfs).groupby(['id', 'prompt', 'model_x', 'model_y', 'left', 'right']).apply(
        partial(aggregate, rng=rng)
    ).reset_index(name='winner')

    df.to_csv(args.output, sep=args.delimiter, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

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

    with nullcontext(subparsers.add_parser('aggregated')) as subparser:
        subparser.add_argument('inputs', type=argparse.FileType('rb'), nargs='?',
                               default=['crowd-comparisons.csv',
                                        'gpt4-crowd-comparisons.csv'])
        subparser.add_argument('-d', '--delimiter', type=str, default=',')
        subparser.add_argument('--seed', type=int, default=0)
        subparser.add_argument('--output', type=argparse.FileType('wb'), default='aggregated-crowd-comparisons.csv')
        subparser.set_defaults(func=aggregated)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
