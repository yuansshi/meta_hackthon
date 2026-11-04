import json

from earshot.actions import ActionLog, verify


def write(log: ActionLog, n: int) -> None:
    for i in range(n):
        log.finish(log.start("create_reminder", {"title": f"thing {i}"}, "device"), True, {"ok": True})


def test_receipts_form_a_chain_that_continues_across_sessions(tmp_path):
    path = tmp_path / "actions" / "dev.jsonl"
    write(ActionLog(path), 2)
    write(ActionLog(path), 2)  # a later session picks the chain up where it ended
    assert verify(path) == (True, 4)


def test_editing_deleting_or_reordering_a_receipt_is_caught(tmp_path):
    path = tmp_path / "dev.jsonl"
    write(ActionLog(path), 4)
    lines = path.read_text().splitlines()

    edited = json.loads(lines[1])
    edited["args"]["title"] = "something else"
    path.write_text("\n".join([lines[0], json.dumps(edited), *lines[2:]]) + "\n")
    assert verify(path) == (False, 2)

    path.write_text("\n".join([lines[0], *lines[2:]]) + "\n")  # one receipt deleted
    assert verify(path) == (False, 2)

    path.write_text("\n".join([lines[1], lines[0], *lines[2:]]) + "\n")  # two swapped
    assert verify(path) == (False, 1)
