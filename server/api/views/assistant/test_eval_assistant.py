# Tests for run_one (eval_assistant.py): the helper that runs the assistant for a
# single eval question and shapes the outcome into a result row.
#
# run_assistant is mocked, so this covers the logic run_one owns — the try/except
# that turns a raising question into an error row instead of aborting the batch, the
# tool columns derived from AgentResult.tool_calls, and the invariant that both
# paths emit every CSV column.

from unittest.mock import MagicMock, patch

import pytest

from api.views.assistant.assistant_types import (
    AgentResult,
    ToolCallExecution,
    ToolCallStatus,
    TurnUsage,
)
from api.views.assistant.eval_assistant import FIELDNAMES, run_one

# TODO: add coverage for main()'s CSV output.

# The two run_assistant outcomes, as patch() kwargs so the same pair can drive both
# the per-path tests and the shared column invariant without restating either setup.
_SUCCEEDS = {
    "return_value": AgentResult(
        output_text="answer",
        response_id="resp-1",
        tool_calls=[
            ToolCallExecution(
                name="search_documents",
                status=ToolCallStatus.OK,
                arguments={"query": "lithium"},
                output="docs",
            ),
            ToolCallExecution(
                name="ask_database",
                status=ToolCallStatus.FAILED,
                arguments={"query": "SELECT"},
                error="bad sql",
            ),
        ],
        # Every field differs between the turns, and no total equals either addend:
        # 100+250=350, 16+64=80, 10+25=35, 4+9=13. Identical per-turn numbers could not
        # distinguish "summed the turns" from "read one turn and ignored the rest".
        # The subset invariants hold too (cached <= input, reasoning <= output), so
        # these rows are also a shape a real run could produce.
        turns=[
            TurnUsage(
                response_id="resp-0",
                input_tokens=100,
                cached_input_tokens=16,
                output_tokens=10,
                reasoning_output_tokens=4,
            ),
            TurnUsage(
                response_id="resp-1",
                input_tokens=250,
                cached_input_tokens=64,
                output_tokens=25,
                reasoning_output_tokens=9,
            ),
        ],
    )
}
_RAISES = {"side_effect": Exception("boom")}


@pytest.mark.parametrize(
    "run_assistant_behavior",
    [pytest.param(_SUCCEEDS, id="success-row"), pytest.param(_RAISES, id="error-row")],
)
def test_run_one_row_carries_every_csv_column(run_assistant_behavior):
    """One invariant over both code paths, so it is parametrized rather than restated.

    This is the only guard on it. csv.DictWriter raises on an *extra* key
    (extrasaction="raise"), which is the direction the FIELDNAMES comment describes —
    but a *missing* key is silently filled with restval (""). So a column added to
    one row literal in run_one and forgotten in the other reaches the CSV as an empty
    cell rather than an error, which is precisely the ragged-row failure FIELDNAMES
    was introduced to prevent.
    """
    with patch(
        "api.views.assistant.eval_assistant.run_assistant", **run_assistant_behavior
    ):
        row = run_one("query", user=MagicMock(), branch="feature")

    assert set(row) == set(FIELDNAMES)


@patch("api.views.assistant.eval_assistant.run_assistant", **_RAISES)
def test_run_one_captures_error(mock_run_assistant):
    row = run_one("query", user=MagicMock(), branch="feature")

    assert row["branch"] == "feature"
    assert row["response_output_text"] is None
    assert "boom" in row["error"]
    # The error row carries every column rather than omitting them, and still records
    # time-to-failure. That the columns are present at all is asserted above; these are
    # their values.
    #
    # All None, not "" or 0. run_assistant raised, but tool calls and turns may already
    # have run and been billed before it did, so these counts are unknown rather than
    # empty. A 0 would be a fake datum: it reads as a run that called no tools and used
    # no tokens, and pandas would average it in. None becomes a blank cell, which pandas
    # treats as missing.
    assert row["tools_called"] is None
    assert row["tool_call_count"] is None
    assert row["tool_error_count"] is None
    assert row["tool_calls_json"] is None
    assert row["turn_count"] is None
    assert row["input_tokens"] is None
    assert row["cached_input_tokens"] is None
    assert row["output_tokens"] is None
    assert row["reasoning_output_tokens"] is None
    assert row["turns_json"] is None
    # Duration is the exception: it was measured, so it is known.
    assert row["duration_s"] > 0


@patch("api.views.assistant.eval_assistant.run_assistant", **_SUCCEEDS)
def test_run_one_records_tool_calls(mock_run_assistant):
    row = run_one("query", user=MagicMock(), branch="feature")

    assert row["tools_called"] == "search_documents|ask_database"
    assert row["tool_call_count"] == 2
    # tool_error_count counts every non-OK status, so one FAILED call stays visible
    # even though the run itself did not raise and `error` is None. That combination
    # is the swallowed-failure hole this column exists to close — a run that reads
    # clean at the row level while a retrieval underneath it broke.
    assert row["tool_error_count"] == 1
    assert row["error"] is None


@patch("api.views.assistant.eval_assistant.run_assistant", **_SUCCEEDS)
def test_run_one_totals_the_turns(mock_run_assistant):
    """The token totals sum every turn, and turn_count counts them.

    Both are derived from result.turns rather than stored, so they cannot disagree
    with each other the way six independent accumulators could.
    """
    row = run_one("query", user=MagicMock(), branch="feature")

    assert row["turn_count"] == 2
    assert row["input_tokens"] == 350
    assert row["cached_input_tokens"] == 80
    assert row["output_tokens"] == 35
    assert row["reasoning_output_tokens"] == 13
    # No total_tokens column: it is input + output, derivable by whoever reads the CSV.
    assert "total_tokens" not in row


@patch("api.views.assistant.eval_assistant.run_assistant")
def test_run_one_totals_are_none_when_any_turn_is_unknown(mock_run_assistant):
    """One turn with unknown usage makes the whole total unknown, not a partial sum.

    This is the guard on the failure mode the whole design is arranged against: a sum
    over only the known turns is indistinguishable in the CSV from a complete one, so
    it would be a real-looking number that isn't real. turn_count stays truthful
    because it counts turns, not tokens.
    """
    mock_run_assistant.return_value = AgentResult(
        output_text="answer",
        response_id="resp-1",
        tool_calls=[],
        turns=[
            TurnUsage(
                response_id="resp-0",
                input_tokens=100,
                cached_input_tokens=16,
                output_tokens=10,
                reasoning_output_tokens=4,
            ),
            # response.usage was missing or an unrecognized shape on this turn.
            TurnUsage(
                response_id="resp-1",
                input_tokens=None,
                cached_input_tokens=None,
                output_tokens=None,
                reasoning_output_tokens=None,
            ),
        ],
    )

    row = run_one("query", user=MagicMock(), branch="feature")

    assert row["turn_count"] == 2
    assert row["input_tokens"] is None
    assert row["cached_input_tokens"] is None
    assert row["output_tokens"] is None
    assert row["reasoning_output_tokens"] is None
