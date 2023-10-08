# Large Language Model Feedback Analysis and Optimization (LLMFAO)

## Datasets

The original [Crowdsourced LLM Benchmark](https://benchmarks.llmonitor.com/) dataset in files `prompts.jsonl`, `models.jsonl`, and `results.jsonl` was kindly provided by the team at [llmonitor.com](https://llmonitor.com/) under a [CC&nbsp;BY 4.0] license.

The derivative files `pairs.jsonl` and `pairs-basic.jsonl` are released under the same [CC&nbsp;BY 4.0] license, as well as `gpt-instruction.txt`, `crowd-instruction.md`, and all other data files in this repository.

## Code

All the code is released under the [GPLv3+] license, but if you use only the data and not the code, it does not apply to you.

```shell
./pairs.py  # generates pairs.jsonl and pairs-basic.jsonl
./requests.py gpt3  # generates gpt3.jsonl (makes no requests to the API, don't worry)
./requests.py gpt4  # generates gpt4.jsonl (makes no requests to the API, don't worry)
```

The generated `gpt3.jsonl` and `gpt4.jsonl` files can be used to make requests to the OpenAI API via [api_request_parallel_processor.py](https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py) (not included here).

```shell
# GPT-3.5 Turbo Instruct
python3 api_request_parallel_processor.py \
    --requests_filepath 'gpt3.jsonl' \
    --save_filepath 'gpt3-responses.jsonl' \
    --request_url 'https://api.openai.com/v1/completions' \
    --max_requests_per_minute 3000 \
    --max_tokens_per_minute 80000
```
```shell
# GPT-4
python3 api_request_parallel_processor.py \
    --requests_filepath 'gpt4.jsonl' \
    --save_filepath 'gpt4-responses.jsonl' \
    --request_url 'https://api.openai.com/v1/chat/completions' \
    --max_requests_per_minute 150 \
    --max_tokens_per_minute 9000
```

[CC&nbsp;BY 4.0]: LICENSE.CC-BY
[GPLv3+]: LICENSE.GPL
