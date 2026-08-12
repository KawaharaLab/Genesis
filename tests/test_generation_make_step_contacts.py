import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generation_make_step", ROOT / "src" / "generation" / "make_step.py"
)
make_step = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(make_step)


class _Link:
    def __init__(self, index):
        self.idx = index


class _Entity:
    def __init__(self, entity_index, link_indices):
        self.idx = entity_index
        self.links = [_Link(index) for index in link_indices]


def test_contact_targets_store_link_indices_not_entity_indices():
    support = _Entity(3, [17, 18])
    obstacle = _Entity(4, [23])

    make_step.set_object_contact_targets([support], obstacle)

    assert make_step.TARGET_SUPPORT_LINK_IDXS == {17, 18}
    assert make_step.OBSTACLE_LINK_IDXS == {23}

