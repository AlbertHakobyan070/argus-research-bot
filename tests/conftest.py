import pytest

from argus import llm


@pytest.fixture(autouse=True)
def _no_quota_latch():
    """Each test starts with no provider latched out of quota (llm._quota_out is process-wide)."""
    llm._quota_out.clear()
    yield
    llm._quota_out.clear()
