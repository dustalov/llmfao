#!/usr/bin/env python3

__author__ = 'Dmitry Ustalov'
__license__ = 'GPL-3.0-or-later'

import argparse
from string import Template
from typing import Dict, List

import pandas as pd


# https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py

def build_metadata(row: pd.Series) -> Dict[str, int]:
    return {
        'id': row['id'],
        'prompt': row['prompt'],
        'model_x': row['model_x'],
        'model_y': row['model_y']
    }


def gpt3(args: argparse.Namespace) -> None:
    df = pd.read_json(args.pairs, lines=True)

    template = Template(args.instruction.read())

    df['metadata'] = df.apply(build_metadata, axis=1)
    df['max_tokens'] = 3
    df['model'] = 'gpt-3.5-turbo-instruct'
    df['prompt'] = df.apply(lambda row: template.safe_substitute(
        text=row['text'].strip(),
        result_x=row['result_x'].strip(),
        result_y=row['result_y'].strip()
    ).strip(), axis=1)

    df = df[['model', 'prompt', 'metadata', 'max_tokens']]

    df.to_json(args.output, orient='records', lines=True)


def gpt4(args: argparse.Namespace) -> None:
    df = pd.read_json(args.pairs, lines=True)

    template = Template(args.instruction.read())

    df['metadata'] = df.apply(build_metadata, axis=1)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=argparse.FileType('rb'), default='pairs.jsonl')
    parser.add_argument('--instruction', type=argparse.FileType('r', encoding='UTF-8'), default='gpt-instruction.txt')

    subparsers = parser.add_subparsers()
    parser_gpt3 = subparsers.add_parser('gpt3')
    parser_gpt3.add_argument('--output', type=argparse.FileType('wb'), default='gpt3.jsonl')
    parser_gpt3.set_defaults(func=gpt3)

    parser_gpt4 = subparsers.add_parser('gpt4')
    parser_gpt4.add_argument('--output', type=argparse.FileType('wb'), default='gpt4.jsonl')
    parser_gpt4.set_defaults(func=gpt4)

    args = parser.parse_args()
    assert hasattr(args, 'func'), 'no command given'
    args.func(args)


if __name__ == '__main__':
    main()
