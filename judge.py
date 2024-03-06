from collections.abc import Generator, Iterable
from dataclasses import dataclass

import solver


@dataclass
class JudgeResult:
    result: solver.Result
    score: float


def judge_results(results: Iterable[solver.Result]) -> Generator[JudgeResult, None, None]:
    """For each result, determine whether the result is good (the two cpc methods agreed) or bad (they disagreed)"""
    # Compare ignoring case
    return (
        JudgeResult(
            result=result,
            score=1.0 if result.gut_check.lower() == result.cot.lower() else 0.0
        )
        for result in results
    )