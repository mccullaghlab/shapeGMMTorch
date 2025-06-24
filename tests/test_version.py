import pytest
import shapeGMMTorch

def test_version_string():
    assert hasattr(shapeGMMTorch, '__version__')
    assert isinstance(shapeGMMTorch.__version__, str)
    assert len(shapeGMMTorch.__version__) > 0

