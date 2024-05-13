import json
import random
from concurrent import futures
from typing import Tuple
import re

import pandas as pd
from tqdm import tqdm, trange

from llm import LLM

"""How good are the models at playing multi-armed bandit games? Does cot prompting beat single-token?"""

gpt35turbo = LLM('gpt-3.5-turbo')
gpt4 = LLM('gpt-4')


class Prompter:
    def __init__(self, llm):
        self.llm = llm

    def prompt(self, history: list, remaining_turns: int = None) -> Tuple[int, list]:
        raise NotImplemented


def game(prompter, num_iterations=50):
    score = 0
    bandits = [50, 100, 150]
    random.shuffle(bandits)
    history = []
    results = []

    for turn_index in trange(num_iterations, desc='running game'):
        remaining_turns = num_iterations - turn_index
        bandit_index, history = prompter.prompt(history, remaining_turns)
        bandit_result = int(random.gauss(bandits[bandit_index], 30))
        history.append({
            'role': 'user',
            'content': f"You pulled bandit {chr(bandit_index + ord('A'))} and got {bandit_result} points."
        })
        score += bandit_result
        results.append({
            'choice': bandit_index,
            'reward': bandit_result,
            'score': score,
            'history': json.dumps(history),
        })

    return bandits, score, pd.DataFrame(results)


class SingleTokenBanditPrompter(Prompter):
    def prompt(self, history, remaining_turns=None):
        history = history or [
            {
                'role': 'user',
                'content': """You are playing a multi-armed-bandit game with three bandits. The mean payout of each bandit is unknown. Get as much score as possible!"""
            }
        ]
        history.append({
            'role': 'user',
            'content': 'Which bandit do you choose? Say A, B, or C:'
        })
        result = self.llm.single_token_completion(history, {'32': 100, '33': 100, '34': 100})
        history.append({
            'role': 'assistant',
            'content': f'{result}'
        })
        choice = ord(result) - ord('A')
        return choice, history


# %%
bandits, score, g = game(SingleTokenBanditPrompter(gpt35turbo), 15)


# %%
# As we increase the number of iterations, does the player more often converge to the correct bandit, and does their
# average score-per-turn increase?
def experiment5_convergence(prompter):
    results = []
    for num_iterations in tqdm([5, 10, 15, 20, 25, 30, 35, 40, 45], desc='running games'):
        bandits, score, g = game(prompter, num_iterations)
        results.append({
            'num_iterations': num_iterations,
            'bandits': bandits,
            'score': score,
            'score_per_turn': score / num_iterations,
            'percent_correct': sum(g['choice'] == bandits.index(max(bandits))) / num_iterations,
            'game': json.dumps(g.to_dict())
        })
    return pd.DataFrame(results)


# %%
convergence_results = experiment5_convergence(SingleTokenBanditPrompter(gpt4))


# %%
# Does the CoT prompter do better?
def filter_thoughts(history):
    # Filter out the llm's thoughts from the history, leaving only its choices, thus compacting the context
    rewritten = []
    for msg in history:
        if msg['role'] == 'assistant':
            try:
                content_object = json.loads(msg['content'])
                rewritten.append({'role': 'assistant', 'content': json.dumps({'choice': content_object['choice']})})
            except json.JSONDecodeError:
                pass
        else:
            rewritten.append(msg)
    return rewritten


class CoTBanditPrompter(Prompter):
    def prompt(self, history, remaining_turns=None):
        history = history or [
            {
                'role': 'user',
                'content': """You are playing a multi-armed-bandit game with three bandits. The mean payout of each bandit is unknown. Get as much score as possible! Make sure to try all three bandits.
                
When choosing a bandit, respond in json:
{
    "thoughts": <string, thinking carefully about which bandit to choose; show your work and think step by step.>
    "choice": "A", "B" or "C"
}"""
            }
        ]
        history.append({
            'role': 'user',
            'content': 'Which bandit do you choose? Respond in json:'
        })
        result = self.llm.chat_completion(filter_thoughts(history))
        history.append({
            'role': 'assistant',
            'content': result
        })
        try:
            result_json = json.loads(result)
        except json.JSONDecodeError as e:
            history.append({'role': 'user', 'content': str(e)})
            return self.prompt(history)
        choice = ord(result_json['choice']) - ord('A')
        return choice, history


# %% Run x games of y rounds each
def experiment5(prompter, num_games=10, game_length=15):
    games = []
    fs = []
    with futures.ThreadPoolExecutor() as executor:
        for _ in range(num_games):
            fs.append(executor.submit(game, prompter, game_length))
    for f in tqdm(futures.as_completed(fs), total=len(fs), desc=f'Running {num_games} games'):
        bandits, score, g = f.result()
        games.append({
            'num_iterations': game_length,
            'bandits': bandits,
            'accuracy': sum(g['choice'] == bandits.index(max(bandits))) / game_length,
            'game': json.dumps(g.to_dict())
        })
    result = pd.DataFrame(games)
    print('average accuracy', result['accuracy'].mean())
    return result


def show_game(df, index=None):
    # if index is none, show the game with the lowest accuracy
    if index is None:
        row = df.loc[df['accuracy'].idxmin()]
    else:
        row = df.iloc[index]
    game_to_show = pd.DataFrame(json.loads(row['game']))
    game_json = game_to_show.iloc[-1]['history']
    print(row['bandits'])
    print('accuracy', row['accuracy'])
    print(json.dumps(json.loads(game_json), indent=2))


# %%
# Run for gpt35t and gpt4, singletoken
experiment5_singletoken_35t = experiment5(SingleTokenBanditPrompter(gpt35turbo), num_games=15)
experiment5_singletoken_4 = experiment5(SingleTokenBanditPrompter(gpt4), num_games=15)

# %%
# average accuracies?
print(experiment5_singletoken_35t['accuracy'].mean())
print(experiment5_singletoken_4['accuracy'].mean())
# save to disk
experiment5_singletoken_35t.to_csv('results/experiment5_singletoken_35t.csv', index=False)
experiment5_singletoken_4.to_csv('results/experiment5_singletoken_4.csv', index=False)

# %% again for gpt4 cot
experiment5_cot_4 = experiment5(CoTBanditPrompter(gpt4), num_games=15)
print(experiment5_cot_4['accuracy'].mean())
experiment5_cot_4.to_csv('results/experiment5_cot_condensedcontext_gpt4.csv', index=False)


# %%
# Let's change our condensation strategy. Instead of filtering out the assistant's thoughts from the context, we'll just
# replace the chat history with a user message summarizing the game's history so far.
# For example, { 'role': 'user', 'content': 'You have played 5 rounds so far. In round 1, you chose bandit A and got
# 100 points. In round 2, you chose bandit B and got 50 points...' }
def history_summary(history):
    # Look for the user messages like "You pulled bandit A and got 100 points."
    summary = []
    for i, msg in enumerate(history):
        if msg['role'] == 'user':
            match = re.match(r'You pulled bandit ([A-C]) and got (\d+) points.', msg['content'])
            if match:
                bandit, points = match.groups()
                summary.append(f'In round {i + 1}, you chose bandit {bandit} and got {points} points.')
    return '\n'.join(summary)


class CoTBanditPrompterSummarization(Prompter):
    def prompt(self, history, remaining_turns=None):
        history = history or [
            {
                'role': 'user',
                'content': """You are playing a multi-armed-bandit game with three bandits. The mean payout of each bandit is unknown. Get as much score as possible! Make sure to try all three bandits.
                
When choosing a bandit, respond in json:
{
    "thoughts": <string, thinking carefully about which bandit to choose; show your work and think step by step.>
    "choice": "A", "B" or "C"
}"""
            }
        ]
        history.append({
            'role': 'user',
            'content': 'Which bandit do you choose? Respond in json:'
        })
        prompt = history[:1] + [{'role': 'user', 'content': history_summary(history)}] + history[-1:]
        result = self.llm.chat_completion(prompt)
        history.append({
            'role': 'assistant',
            'content': result
        })
        try:
            result_json = json.loads(result)
            choice = ord(result_json['choice']) - ord('A')
        except (json.JSONDecodeError, KeyError) as e:
            history.append({'role': 'user', 'content': str(e)})
            return self.prompt(history)
        return choice, history


# %% run it for 4 and 3.5
experiment5_cot_35t = experiment5(CoTBanditPrompterSummarization(gpt35turbo))
experiment5_cot_4 = experiment5(CoTBanditPrompterSummarization(gpt4))
# %% save to disk
experiment5_cot_35t.to_csv('results/experiment5_cot_summarized_gpt35t.csv', index=False)
experiment5_cot_4.to_csv('results/experiment5_cot_summarized_gpt4.csv', index=False)


# %% What if we tell the llm how many turns are remaining?
def history_summary_with_remaining_turns(history, remaining_turns):
    summary = history_summary(history)
    return f'{summary}\nYou have {remaining_turns} turns remaining.'


class CoTBanditPrompterSummarizationRemainingTurns(Prompter):
    def prompt(self, history, remaining_turns=None):
        history = history or [
            {
                'role': 'user',
                'content': """You are playing a multi-armed-bandit game with three bandits. The mean payout of each bandit is unknown. Get as much score as possible! Make sure to try all three bandits.
                
When choosing a bandit, respond in json:
{
    "thoughts": <string, thinking carefully about which bandit to choose; show your work and think step by step.>
    "choice": "A", "B" or "C"
}"""
            }
        ]
        prompt = history[:1] + [
            {'role': 'user', 'content': history_summary_with_remaining_turns(history, remaining_turns)}] + [
                     {'role': 'user', 'content': 'Which bandit do you choose? Respond in json:'}]
        result = self.llm.chat_completion(prompt)
        history.append({
            'role': 'assistant',
            'content': result
        })
        try:
            result_json = json.loads(result)
            choice = ord(result_json['choice']) - ord('A')
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            history.append({'role': 'user', 'content': str(e)})
            return self.prompt(history)
        return choice, history


# %% run it for 4 and 3.5
experiment5_cot_35t = experiment5(CoTBanditPrompterSummarizationRemainingTurns(gpt35turbo), num_games=15)
experiment5_cot_4 = experiment5(CoTBanditPrompterSummarizationRemainingTurns(gpt4), num_games=15)
experiment5_cot_35t.to_csv('results/experiment5/experiment5_cot_summarized_remainingturns_gpt35t.csv', index=False)
experiment5_cot_4.to_csv('results/experiment5/experiment5_cot_summarized_remainingturns_gpt4.csv', index=False)

# %% try gpt4turbo
gpt4turbo = LLM('gpt-4-turbo')
experiment5_cot_4turbo = experiment5(CoTBanditPrompterSummarizationRemainingTurns(gpt4turbo))
experiment5_cot_4turbo.to_csv('results/experiment5/experiment5_cot_summarized_remainingturns_gpt4turbo.csv',
                              index=False)


# %%
# Single-token version of CoTBanditPrompterSummarizationRemainingTurns
class SingleTokenBanditPrompterSummarizationRemainingTurns(Prompter):
    def prompt(self, history, remaining_turns=None):
        history = history or [
            {
                'role': 'user',
                'content': """You are playing a multi-armed-bandit game with three bandits. The mean payout of each bandit is unknown. Get as much score as possible! Make sure to try all three bandits."""
            }
        ]
        history.append({
            'role': 'user',
            'content': 'Which bandit do you choose? Say A, B, or C:'
        })
        prompt = history[:1] + [
            {'role': 'user', 'content': history_summary_with_remaining_turns(history, remaining_turns)}] + [
                     {'role': 'user', 'content': 'Which bandit do you choose? Say A, B, or C:'}]
        print(prompt)
        result = self.llm.single_token_completion(prompt, {'32': 100, '33': 100, '34': 100})
        history.append({
            'role': 'assistant',
            'content': f'{result}'
        })
        choice = ord(result) - ord('A')
        return choice, history

# %%
# run it for 4 and 3.5
experiment5_singletoken_35t = experiment5(SingleTokenBanditPrompterSummarizationRemainingTurns(gpt35turbo), num_games=1)
experiment5_singletoken_4 = experiment5(SingleTokenBanditPrompterSummarizationRemainingTurns(gpt4), num_games=15)
experiment5_singletoken_35t.to_csv('results/experiment5/experiment5_singletoken_summarized_remainingturns_gpt35t.csv', index=False)
experiment5_singletoken_4.to_csv('results/experiment5/experiment5_singletoken_summarized_remainingturns_gpt4.csv', index=False)
