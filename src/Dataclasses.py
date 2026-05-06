from typing import Literal, Optional, Any, List, Tuple, Dict
from dataclasses import dataclass
from collections import Counter

@dataclass
class Node:
   """
      Dataclass for Node in the tree.
   """
   type: Literal["leaf", "internal"]
   distribution: Counter[Any]
   label: Optional[str] = None   # class
   samples: Optional[int] = None
   feature: Optional[str] = None
   gain_ratio: Optional[float] = None
   information_gain: Optional[float] = None
   threshold: Optional[float] = None
   # Children
   left: Optional["Node"] = None
   right: Optional["Node"] = None


@dataclass
class DiagnosticOutput:
   """
      Comprehensive diagnostic output for a single prediction.
   """

   # Basic Prediction
   predicted_class: str
   confidence: float

   # Decision Path
   decision_path: List[Tuple[str, float, str]]  # [(feature, threshold, direction), ...]
   decision_path_readable: str

   # Domain-level analysis
   domain_severity_scores: Dict[str, float]

   # Task-level analysis
   task_importance_scores: Dict[str, float]

   # Raw leaf distribution
   leaf_distribution: Dict[Any, int]

