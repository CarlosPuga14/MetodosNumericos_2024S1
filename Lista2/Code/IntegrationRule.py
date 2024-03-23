"""
The base class for numerical integration.
This is an abstract class that defines the interface for numerical integration.
Integration Methods are supposed to inherit from this class and implement the integration rule.

Created by Carlos Puga - 03/23/2024
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class Integration(metaclass=ABC):
    pass