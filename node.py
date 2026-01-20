from typing import Literal, Optional, Any
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
    threshold: Optional[float] = None
    # Children
    left: Optional["Node"] = None
    right: Optional["Node"] = None
