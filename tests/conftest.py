import pytest

from pv_agent.synthetic import build_synthetic_db
from pv_agent.tools import PharmacovigilanceTools


@pytest.fixture(scope="session")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "test.db"
    build_synthetic_db(path, n_cases=6000, seed=7)
    return path


@pytest.fixture()
def tools(db_path):
    return PharmacovigilanceTools(db_path)
