import json
import random
import re
from concurrent import futures
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm, trange

from llm import LLM

"""How good are the models at playing multi-armed bandit games? Does cot prompting beat single-token?"""

gpt35turbo = LLM('gpt-3.5-turbo')
gpt4 = LLM('gpt-4')
gpt4t = LLM('gpt-4-turbo')
gpt4o = LLM('gpt-4o')


class Prompter:
    def __init__(self, llm):
        self.llm = llm

    def prompt(self, history: list, remaining_turns: int = None) -> Tuple[int, list]:
        raise NotImplemented


class Bandit:
    def pull(self) -> int:
        raise NotImplemented

    # In comparisons, the bandit with the highest EV is the largest bandit
    def __lt__(self, other):
        raise NotImplemented

    def to_dict(self):
        raise NotImplemented


class GaussianBandit(Bandit):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def pull(self) -> int:
        return int(random.gauss(self.mean, self.std))

    def __str__(self):
        return f'GaussianBandit(mean={self.mean}, std={self.std})'

    def __lt__(self, other):
        return self.mean < other.mean

    def to_dict(self):
        return {'mean': self.mean, 'std': self.std}


def default_bandits():
    bandits = [GaussianBandit(50, 30), GaussianBandit(100, 30), GaussianBandit(150, 30)]
    return random.sample(bandits, len(bandits))


def game(prompter, bandits, num_turns=20, progress_bar=None):
    history = []
    results = []

    for turn_index in range(num_turns):
        remaining_turns = num_turns - turn_index
        bandit_index, history = prompter.prompt(history, remaining_turns)
        bandit_result = bandits[bandit_index].pull()
        history.append({
            'role': 'user',
            'content': f'You pulled bandit {chr(bandit_index + ord("A"))} and got {bandit_result} points.'
        })
        results.append({
            'bandits': bandits,
            'choice': bandit_index,
            'reward': bandit_result,
            'history': json.dumps(history),
        })

        if progress_bar:
            progress_bar.update(1)

    if progress_bar:
        progress_bar.close()
    return pd.DataFrame(results)


class SingleTokenBanditPrompter(Prompter):
    def prompt(self, history, remaining_turns=0):
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
# As we increase the number of iterations, does the player more often converge to the correct bandit, and does their
# average score-per-turn increase?
def experiment5_convergence(prompter):
    results = []
    for num_turns in tqdm([5, 10, 15, 20, 25, 30, 35, 40, 45], desc='running games'):
        bandits = default_bandits()
        g = game(prompter, bandits, num_turns)
        results.append({
            'num_iterations': num_turns,
            'bandits': bandits,
            'accuracy': sum(g['choice'] == bandits.index(max(bandits))) / num_turns,
            'game': json.dumps(g.to_dict(), default=vars)
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
        result = self.llm.chat_completion(filter_thoughts(history), json=True)
        history.append({
            'role': 'assistant',
            'content': result
        })
        try:
            result_json = json.loads(result)
        except json.JSONDecodeError as e:
            history.append({'role': 'user', 'content': str(e)})
            return self.prompt(history, remaining_turns)
        choice = ord(result_json['choice']) - ord('A')
        return choice, history


# %% Run x games of y rounds each
def experiment5(prompter, num_games=10, game_length=15):
    games = []
    fs = []
    with futures.ThreadPoolExecutor() as executor:
        for i in range(num_games):
            progress_bar = tqdm(total=game_length, desc=f"Game {i + 1}", position=i)
            fs.append(executor.submit(game, prompter, default_bandits(), game_length, progress_bar))
    for f in futures.as_completed(fs):
        g = f.result()
        bandits = g.iloc[0]['bandits']
        games.append({
            'num_iterations': game_length,
            'accuracy': sum(g['choice'] == bandits.index(max(bandits))) / game_length,
            'game': json.dumps(g.to_dict(), default=vars)
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
    print(chr(game_to_show.iloc[-1]['choice'] + ord('A')), game_to_show['bandits'])
    game_json = game_to_show.iloc[-1]['history']
    print('accuracy', row['accuracy'])
    print(json.dumps(json.loads(game_json), indent=2))


# %%
# Run for gpt35t and gpt4o, singletoken
experiment5_singletoken_35t = experiment5(SingleTokenBanditPrompter(gpt35turbo), num_games=1)
experiment5_singletoken_4o = experiment5(SingleTokenBanditPrompter(gpt4o), num_games=15)
experiment5_singletoken_35t.to_csv('results/experiment5_singletoken_35t.csv', index=False)
experiment5_singletoken_4o.to_csv('results/experiment5_singletoken_4o.csv', index=False)

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
    turn_count = 1
    for _, msg in enumerate(history):
        if msg['role'] == 'user':
            match = re.match(r'You pulled bandit ([A-C]) and got -?(\d+) points.', msg['content'])
            if match:
                bandit, points = match.groups()
                summary.append(f'In round {turn_count}, you chose bandit {bandit} and got {points} points.')
                turn_count += 1
    if turn_count > 1:
        return '\n'.join(summary)
    else:
        return 'This is your first turn and you have not made any pulls yet.'


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
        result = self.llm.chat_completion(prompt, json=True)
        history.append({
            'role': 'assistant',
            'content': result
        })
        try:
            result_json = json.loads(result)
            choice = ord(result_json['choice']) - ord('A')
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(result)
            history.append({'role': 'user', 'content': str(e)})
            return self.prompt(history, remaining_turns)
        return choice, history


# %% run it for 4 and 3.5
experiment5_cot_35t = experiment5(CoTBanditPrompterSummarization(gpt35turbo))
experiment5_cot_4o = experiment5(CoTBanditPrompterSummarization(gpt4o))
# %% save to disk
experiment5_cot_35t.to_csv('results/experiment5_cot_summarized_gpt35t.csv', index=False)
experiment5_cot_4o.to_csv('results/experiment5_cot_summarized_gpt4o.csv', index=False)


# %% What if we tell the llm how many turns are remaining?
def history_summary_with_remaining_turns(history, remaining_turns):
    summary = history_summary(history)
    return f'{summary}\nYou have {remaining_turns} pull(s) remaining.'


class CoTBanditPrompterSummarizationRemainingTurns(Prompter):
    def prompt(self, history, remaining_turns=0):
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
        result = self.llm.chat_completion(prompt, json=True)
        history.append({
            'role': 'assistant',
            'content': result
        })
        try:
            result_json = json.loads(result)
            choice = ord(result_json['choice']) - ord('A')
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            history.append({'role': 'user', 'content': str(e)})
            return self.prompt(history, remaining_turns)
        return choice, history


# %% run it for various models
experiment5_cot_35t = experiment5(CoTBanditPrompterSummarizationRemainingTurns(gpt35turbo), num_games=15)
experiment5_cot_4t = experiment5(CoTBanditPrompterSummarizationRemainingTurns(gpt4t), num_games=15)
experiment5_cot_4 = experiment5(CoTBanditPrompterSummarizationRemainingTurns(gpt4), num_games=15)
experiment5_cot_4o = experiment5(CoTBanditPrompterSummarizationRemainingTurns(gpt4o), num_games=15)
experiment5_cot_35t.to_csv('results/experiment5/experiment5_cot_summarized_remainingturns_gpt35t.csv', index=False)
experiment5_cot_4.to_csv('results/experiment5/experiment5_cot_summarized_remainingturns_gpt4.csv', index=False)
experiment5_cot_4o.to_csv('results/experiment5/experiment5_cot_summarized_remainingturns_gpt4o.csv', index=False)
experiment5_cot_4t.to_csv('results/experiment5/experiment5_cot_summarized_remainingturns_gpt4t.csv', index=False)

# %% Let's try the single-token edition
class SingleTokenBanditPrompterSummarizationRemainingTurns(Prompter):
    def prompt(self, history, remaining_turns=0):
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
        result = self.llm.single_token_completion(prompt, {'32': 100, '33': 100, '34': 100})
        history.append({
            'role': 'assistant',
            'content': f'{result}'
        })
        choice = ord(result) - ord('A')
        return choice, history


# %%
# run it for 4o and 3.5
experiment5_singletoken_35t = experiment5(SingleTokenBanditPrompterSummarizationRemainingTurns(gpt35turbo))
experiment5_singletoken_4o = experiment5(SingleTokenBanditPrompterSummarizationRemainingTurns(gpt4o))
experiment5_singletoken_4 = experiment5(SingleTokenBanditPrompterSummarizationRemainingTurns(gpt4))
experiment5_singletoken_35t.to_csv('results/experiment5/experiment5_singletoken_summarized_remainingturns_gpt35t.csv',
                                   index=False)
experiment5_singletoken_4o.to_csv('results/experiment5/experiment5_singletoken_summarized_remainingturns_gpt4o.csv',
                                 index=False)


# %% Curious that for CoT, 35t is beating 4. This seems to be because 35t is more exploitative
# and 4 is more exploratory. In that case I would expect the gap to invert as the bandits become more similar.
# Is this the case?
def experiment_bandits_variance_vs_accuracy(prompter, num_games=20, game_length=15):
    games = []
    fs = []
    with futures.ThreadPoolExecutor() as executor:
        for _ in range(num_games):
            deviation = random.randint(30, 50)
            bandits = [
                GaussianBandit(50, deviation),
                GaussianBandit(100, deviation),
                GaussianBandit(150, deviation)
            ]
            fs.append(executor.submit(game, prompter, bandits, game_length, None))
    for f in tqdm(futures.as_completed(fs), total=len(fs)):
        g = f.result()
        bandits = g.iloc[0]['bandits']
        games.append({
            'standard_deviation': bandits[0].std,
            'accuracy': sum(g['choice'] == bandits.index(max(bandits))) / game_length,
            'game': json.dumps(g.to_dict(), default=vars)
        })
    result = pd.DataFrame(games)
    return result


# %%
bandits_variance_vs_accuracy_35t = experiment_bandits_variance_vs_accuracy(
    CoTBanditPrompterSummarizationRemainingTurns(gpt35turbo))
bandits_variance_vs_accuracy_4 = experiment_bandits_variance_vs_accuracy(
    CoTBanditPrompterSummarizationRemainingTurns(gpt4))
bandits_variance_vs_accuracy_35t['model'] = 'gpt-3.5-turbo'
bandits_variance_vs_accuracy_4['model'] = 'gpt-4'
bandits_variance_vs_accuracy = pd.concat([bandits_variance_vs_accuracy_35t, bandits_variance_vs_accuracy_4])
bandits_variance_vs_accuracy.to_csv('results/experiment5/bandits_variance_vs_accuracy.csv', index=False)

# %%
# Analyze variance vs accuracy. How strongly do they correlate? Is there a difference between the models?
sns.lmplot(data=bandits_variance_vs_accuracy, x='standard_deviation', y='accuracy', hue='model')
plt.show()

# %%
# What if we tell it to use UCB?
UCB_explanation = """The Upper Confidence Bound algorithm is as follows.
Step 1. For the first 3 rounds, choose each arm once. This is the initial exploration phase.
Step 2. After the first K rounds, for each subsequent round t, select the arm j that maximizes:
(average reward of arm j) + sqrt((2 * ln(t)) / (number of times arm j has been selected))
where:
"average reward of arm j" is the average reward obtained from arm j so far
"number of times arm j has been selected" is the number of times arm j has been selected so far
t is the total number of rounds played so far
ln(t) is the natural logarithm of t
sqrt(x) is the square root of x

Repeat step 2 until the horizon is reached."""


# This prompter prompts the llm to use UCB, including the explanation of the algorithm
class CoTBanditPrompterUCB(Prompter):
    def prompt(self, history, remaining_turns=0):
        history = history or [
            {
                'role': 'user',
                'content': """\nYou are playing a multi-armed-bandit game with three bandits. Each arm returns from a gaussian distribution with unknown mean and standard deviation. It's optimal therefore to use the Upper Confidence Bound algorithm.\n"""
                + UCB_explanation + """\nRespond in json like this:
{
    "thoughts": <string. Here you will think step by step to using the UCB algorithm. After the exploration phase, you must calculate a UCB value for each arm here in this "thoughts" block.>
    "choice": "A", "B" or "C"
}"""
            }
        ]
        summary = {'role': 'user', 'content': history_summary_with_remaining_turns(history, remaining_turns)}
        history.append(summary)
        prompt = history[:1] + [summary] + [{'role': 'user', 'content': 'Which bandit do you choose? Begin your answer with a left curly brace ({):'}]
        result = self.llm.chat_completion(prompt, json=True)
        history.append({
            'role': 'assistant',
            'content': result
        })
        try:
            result_json = json.loads(result)
            choice = ord(result_json['choice']) - ord('A')
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(result)
            history.append({'role': 'user', 'content': str(e)})
            return self.prompt(history, remaining_turns)
        return choice, history


# %% run it for 3.5t and 4 and 3.5
experiment5_cot_ucb_35t = experiment5(CoTBanditPrompterUCB(gpt35turbo))
experiment5_cot_ucb_4o = experiment5(CoTBanditPrompterUCB(gpt4o))
experiment5_cot_ucb_35t.to_csv('results/experiment5/experiment5_cot_ucb_gpt35t.csv', index=False)
experiment5_cot_ucb_4o.to_csv('results/experiment5/experiment5_cot_ucb_gpt4o.csv', index=False)

# %%
# How do the different models (35t, 4t, 4o) perform across four algorithms (intuitive, random, greedy, and UCB)
# To mitigate prompt sensitivity, three prompt variations are used
