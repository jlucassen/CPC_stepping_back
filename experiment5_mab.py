import json
import random
from typing import Tuple
from concurrent import futures

import pandas as pd
from tqdm import tqdm, trange

from llm import LLM

"""How good are the models at playing multi-armed bandit games? Does cot prompting beat single-token?"""

gpt35turbo = LLM('gpt-3.5-turbo')
gpt4 = LLM('gpt-4')


class Prompter:
    def __init__(self, llm):
        self.llm = llm

    def prompt(self, history: list) -> Tuple[int, list]:
        raise NotImplemented


def game(prompter, num_iterations=50):
    score = 0
    bandits = [50, 100, 150]
    random.shuffle(bandits)
    history = []
    results = []

    for _ in trange(num_iterations, desc='running game'):
        bandit_index, history = prompter.prompt(history)
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
    def prompt(self, history):
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
    def prompt(self, history):
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


# %%
def experiment5(prompter, num_games=5, game_length=20):
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
            'score': score,
            'score_per_turn': score / game_length,
            'percent_correct': sum(g['choice'] == bandits.index(max(bandits))) / game_length,
            'game': json.dumps(g.to_dict())
        })
    return pd.DataFrame(games)


games_gpt4 = experiment5(CoTBanditPrompter(gpt4))
result_game = pd.DataFrame(json.loads(games_gpt4.iloc[4]['game']))
record = json.loads(result_game.iloc[-1]['history'])
