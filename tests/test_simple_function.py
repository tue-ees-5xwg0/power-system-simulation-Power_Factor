from power_system_simulation.assignment_1 import GraphProcessor
from power_system_simulation.simple_function import add, multiply


def test_add():
    """Check if adding function works"""
    assert add(1, 1) == 2


def test_multiply():
    """Check if multiply function works"""
    assert multiply(2, 2) == 4
