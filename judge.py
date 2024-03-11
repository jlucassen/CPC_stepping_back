from dataclasses import dataclass
from typing import Any


@dataclass
class JudgeResult:
    result: Any
    score: float
