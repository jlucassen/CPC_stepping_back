import dspy  # pip install dspy-ai
import dspy.evaluate
import pandas as pd
from dotenv import load_dotenv
from dspy.teleprompt import BootstrapFewShot

load_dotenv()
turbo = dspy.OpenAI(model='gpt-3.5-turbo')
dspy.settings.configure(lm=turbo)


class ShouldChangeApproach(dspy.Signature):
    """
    At this point, is the current approach working, or should we change to another approach?
    """

    context = dspy.InputField(desc='The problem-solving narrative')
    answer = dspy.OutputField(desc='One-word answer: "Yes" or "No".')


class CPC(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.Predict(ShouldChangeApproach)

    def forward(self, context):
        prediction = self.generate_answer(context=context)
        return dspy.Prediction(context=context, answer=prediction.answer)


class CoTCPC(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(ShouldChangeApproach)

    def forward(self, context):
        prediction = self.generate_answer(context=context)
        return dspy.Prediction(context=context, answer=prediction.answer)


# split into training and validation sets (90/10)
def to_example(row):
    return dspy.Example(context=row['context'], answer='Yes' if row['is_factorizable'] == 'No' else 'Yes').with_inputs(
        'context')


# create 'train' and 'dev' dataset from results/experiment2_gpt3.csv
dataset = pd.read_csv('results/experiment2_gpt3.csv')
training_set = dataset.sample(frac=0.9).apply(to_example, axis=1)
validation_set = dataset.drop(training_set.index).apply(to_example, axis=1)


def validate(example, pred, trace=None):
    answer_em = dspy.evaluate.answer_exact_match(example, pred)
    answer_pm = dspy.evaluate.answer_passage_match(example, pred)
    return answer_em and answer_pm


teleprompter = BootstrapFewShot(metric=validate)

# %%
compiled = teleprompter.compile(CPC(), trainset=training_set.sample(100).to_list())
