import pytest

from prospecta.stages import parse_json_object

OBJ = '{"correct": true, "why": "a {brace} here"}'
WANT = {"correct": True, "why": "a {brace} here"}


@pytest.mark.parametrize("reply", [
    OBJ,
    OBJ + "\n" + OBJ,
    f"```json\n{OBJ}\n```\n\n```json\n{OBJ}\n```",
    f"Here you go:\n{OBJ}\nAs I said:\n{OBJ}\nHope that helps.",
    f"Sure {{not json}} first.\n```json\n{OBJ}\n```",
    f'{OBJ}\n{{"correct": false}}',
])
def test_first_object_wins(reply):
    assert parse_json_object(reply) == WANT


@pytest.mark.parametrize("reply", ["no json", "[1, 2]", "{broken"])
def test_no_object_raises(reply):
    with pytest.raises(ValueError):
        parse_json_object(reply)
