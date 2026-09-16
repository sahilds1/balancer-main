# Tests for the assistant's tools and the agentic reasoning loop.
#
# Covers the logic these modules own, with mocked collaborators (no DB, no OpenAI):
#   - Tool instances: SEARCH_TOOL.run adapts the loop's uniform (user, **arguments)
#     call into search_documents' own (query, user) signature; schema() emits the
#     flattened Responses-API shape rather than the nested Chat-Completions one.
#   - search_documents' error/empty contract: failing raises, matching nothing does not.
#   - handle_tool_calls: dispatching the model's function calls — the call/no-call
#     branch, output shaping, and both error outcomes. Tools are indexed by name and
#     invoked as tool.run(user, **arguments).
#   - run_agentic_loop: the while-loop that keeps calling the model until it stops
#     emitting tool calls, including loop continuity via previous_response_id and the
#     function_call_output payload fed back on each continuation.
#   - _turn_usage: reading token counts off one response, and the guard that turns an
#     unrecognized usage shape into None rather than into a wrong number. This is the
#     only part of the suite that builds its input with the SDK's own model instead of
#     a mock, for the reason given at _make_usage.
#
# Two tests were removed as glue. test_ask_database_tool_run_ignores_user asserted a
# single-argument forward whose wrong version raises TypeError on first call, and
# test_tools_registry_contains_both_tools restated the TOOLS list literal — a
# change-detector that made "adding a tool is appending one Tool to TOOLS; nothing
# else changes" (tool_services.py) false, since the intended way to extend the code
# was also the way to break the test.
#
# Where two tests were the same test with one input changed, they are now one
# pytest.mark.parametrize case table. Tests whose assertions differ in kind are left
# separate on purpose: folding those together needs a column per optional assertion
# and a body full of conditionals, which costs more clarity than the duplication did.

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from openai.types.responses import ResponseUsage
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

# TODO: add coverage for search_documents' formatting of embeddings results — the
# [Document N - File: ..., Similarity: ...] shape and the multi-result join. No DB
# needed: search_documents only calls get_closest_embeddings and convert_uuids, so
# mocking those two (like the rest of the suite mocks collaborators) is enough. The
# empty-results and exception paths are covered below.
#
# Sequence this after the file_id removal queued in search_tool.py, not before: that
# format string is about to lose its `File: {file_id}` field, so a test written against
# today's shape would be red on arrival. Pinning the format is worth doing either way —
# the field is there because the model reads it, and a change-detector objection doesn't
# apply to output whose exact text is the contract with the model.

from api.views.assistant.assistant_types import (
    AgentResult,
    Tool,
    ToolCallExecution,
    ToolCallStatus,
    TurnUsage,
)
from api.views.assistant.agentic_loop import (
    _turn_usage,
    handle_tool_calls,
    run_agentic_loop,
)
from api.views.assistant.search_tool import search_documents
from api.views.assistant.tool_services import SEARCH_TOOL


# ---------------------------------------------------------------------------
# Response / tool builders
#
# Defined before the tests because pytest.mark.parametrize case tables are built at
# import time, so anything they construct must already exist.
# ---------------------------------------------------------------------------

def _make_function_call_item(name, arguments, call_id):
    item = MagicMock()
    item.type = "function_call"
    item.name = name
    item.arguments = json.dumps(arguments)
    item.call_id = call_id
    return item


def _make_reasoning_item(summary="reasoning summary"):
    item = MagicMock()
    item.type = "reasoning"
    item.summary = summary
    return item


def _make_response(output_items):
    response = MagicMock()
    response.output = output_items
    return response


def _make_terminal_response(output_text, response_id):
    """A response with no function calls — terminates the loop."""
    response = MagicMock()
    response.output = []
    response.output_text = output_text
    response.id = response_id
    return response


def _make_tool_call_response(response_id, query="lithium"):
    """A response with one function call — continues the loop."""
    response = MagicMock()
    response.output = [_make_function_call_item("search_documents", {"query": query}, "call-loop")]
    response.id = response_id
    return response


def _make_client(*responses):
    """A client whose successive responses.create calls return `responses` in order.

    side_effect rather than return_value on purpose: return_value would hand the same
    terminal response back forever, so a loop that failed to terminate would hang or
    silently pass. A list runs out, and the extra call raises StopIteration.
    """
    client = MagicMock()
    client.responses.create.side_effect = list(responses)
    return client


def _fake_tool(name, run):
    """A Tool whose run is a mock; description/parameters are irrelevant to dispatch."""
    return Tool(name=name, description="", parameters={}, run=run)


# ---------------------------------------------------------------------------
# Tool instances
# ---------------------------------------------------------------------------

@patch("api.views.assistant.tool_services.search_documents")
def test_search_tool_run_forwards_query_and_user(mock_search):
    """The adapter inverts the argument order, which is why this is worth asserting.

    The loop calls run(user=..., query=...); search_documents takes (query, user).
    Getting the swap wrong searches with a User object as the query string and scopes
    access control to a string — silent in both directions, and this is the leg where
    document access control is actually enforced.
    """
    mock_search.return_value = "results"
    user = MagicMock()

    SEARCH_TOOL.run(user=user, query="lithium")

    mock_search.assert_called_once_with("lithium", user)


def test_tool_schema_is_flattened_shape():
    schema = SEARCH_TOOL.schema()
    assert schema["type"] == "function"
    assert schema["name"] == "search_documents"
    assert "parameters" in schema
    # The load-bearing assertion: this repo contains both tool-schema shapes, and
    # services/tools/tools.py's create_tool_dict builds the nested Chat-Completions
    # one. The Responses API needs the flattened form, so a copy-paste from there
    # would be accepted by every other assertion here.
    assert "function" not in schema


# ---------------------------------------------------------------------------
# search_documents error/empty contract
#
# These two lock in the distinction the tool's status reporting depends on: a
# retrieval that *fails* must raise (so the loop records FAILED), while a retrieval
# that legitimately *matches nothing* must return normally (so it stays OK). Both
# used to return a string, which made the two indistinguishable downstream.
# ---------------------------------------------------------------------------

@patch("api.views.assistant.search_tool.get_closest_embeddings")
def test_search_documents_raises_instead_of_returning_the_error(mock_get):
    mock_get.side_effect = RuntimeError("embedding backend down")

    # Must propagate. Swallowing it here would report a failed retrieval as a
    # successful tool call and leave ToolCallStatus.FAILED unreachable for this tool.
    with pytest.raises(RuntimeError, match="embedding backend down"):
        search_documents("lithium", user=MagicMock())


@patch("api.views.assistant.search_tool.convert_uuids", return_value=[])
@patch("api.views.assistant.search_tool.get_closest_embeddings", return_value=[])
def test_search_documents_returns_message_when_nothing_matches(mock_get, mock_convert):
    result = search_documents("lithium", user=MagicMock())

    # No match is an outcome, not an error — returns normally so the call records OK.
    assert "No relevant documents found" in result


@patch(
    "api.views.assistant.search_tool.get_closest_embeddings",
    side_effect=RuntimeError("embedding backend down"),
)
def test_failed_status_is_reachable_through_the_real_search_tool(mock_get):
    """The two fixes composed: a real retrieval failure arrives at the eval as FAILED.

    Deliberately dispatches the *real* SEARCH_TOOL — only its embedding dependency is
    mocked — rather than a fake tool that raises. A fake would exercise the identical
    loop branch as test_handle_tool_calls_records_the_two_error_outcomes below and
    prove nothing
    extra; what is worth testing is that search_documents' decision not to swallow the
    exception and the loop's decision to record FAILED actually meet, with the real
    adapter between them.
    """
    item = _make_function_call_item("search_documents", {"query": "lithium"}, "call-e2e")

    _, calls = handle_tool_calls(
        _make_response([item]), tools=[SEARCH_TOOL], user=MagicMock()
    )

    assert calls[0].status is ToolCallStatus.FAILED
    assert "embedding backend down" in calls[0].error


# ---------------------------------------------------------------------------
# handle_tool_calls tests
# ---------------------------------------------------------------------------

def test_handle_tool_calls_returns_empty_lists_when_no_function_calls():
    response = _make_response([_make_reasoning_item()])
    messages, calls = handle_tool_calls(response, tools=[], user=MagicMock())
    assert messages == []
    assert calls == []


def test_handle_tool_calls_dispatches_tool_and_returns_output():
    mock_run = MagicMock(return_value="search result")
    tool = _fake_tool("search_documents", mock_run)
    user = MagicMock()
    item = _make_function_call_item("search_documents", {"query": "lithium"}, "call-1")
    response = _make_response([item])

    messages, calls = handle_tool_calls(response, tools=[tool], user=user)

    # The loop binds user at dispatch and forwards the model's arguments.
    mock_run.assert_called_once_with(user=user, query="lithium")
    # The OpenAI payload (unchanged shape) is the first return value.
    assert messages == [
        {"type": "function_call_output", "call_id": "call-1", "output": "search result"}
    ]
    # The ToolCallExecution record captures the outcome, the model's query, and output.
    assert calls == [
        ToolCallExecution(
            name="search_documents",
            status=ToolCallStatus.OK,
            arguments={"query": "lithium"},
            output="search result",
        )
    ]


@pytest.mark.parametrize(
    "tools, expected_output_fragment, expected_status, expected_error_fragment, expected_arguments",
    [
        pytest.param(
            [],
            "ERROR - No tool registered",
            ToolCallStatus.UNREGISTERED,
            "No tool registered",
            None,
            id="model-named-a-tool-we-do-not-have",
        ),
        pytest.param(
            [_fake_tool("search_documents", MagicMock(side_effect=Exception("tool exploded")))],
            "Error executing function call",
            ToolCallStatus.FAILED,
            "tool exploded",
            {"query": "x"},
            id="registered-tool-raised",
        ),
    ],
)
def test_handle_tool_calls_records_the_two_error_outcomes(
    tools,
    expected_output_fragment,
    expected_status,
    expected_error_fragment,
    expected_arguments,
):
    """FAILED vs UNREGISTERED, parametrized to keep the contrast readable.

    These are opposite diagnoses — a code or data fault on our side vs. the model
    hallucinating a tool name — which is why ToolCallStatus is a three-state enum and
    not a bool, and why a tool-selection eval has to tell them apart.

    Reading them as one table also surfaces a difference neither test stated when they
    were separate: `arguments` is parsed inside the registered branch, so an
    unregistered call records None while a raising tool still reports the query the
    model generated.
    """
    item = _make_function_call_item("search_documents", {"query": "x"}, "call-err")

    messages, calls = handle_tool_calls(
        _make_response([item]), tools=tools, user=MagicMock()
    )

    # Either way the model still gets a message back, so it can retry or say it could
    # not retrieve anything — the loop does not abandon the turn.
    assert messages[0]["call_id"] == "call-err"
    assert expected_output_fragment in messages[0]["output"]

    assert calls[0].name == "search_documents"
    assert calls[0].status is expected_status
    assert expected_error_fragment in calls[0].error
    assert calls[0].arguments == expected_arguments


def test_handle_tool_calls_handles_multiple_calls_in_one_response():
    mock_run = MagicMock(return_value="result")
    tool = _fake_tool("search_documents", mock_run)
    items = [
        _make_function_call_item("search_documents", {"query": "q1"}, "call-4"),
        _make_function_call_item("search_documents", {"query": "q2"}, "call-5"),
    ]
    response = _make_response(items)

    messages, calls = handle_tool_calls(response, tools=[tool], user=MagicMock())

    # Two calls in one response accumulate rather than overwrite — distinct from the
    # cross-iteration accumulation covered in the loop test below.
    assert [m["call_id"] for m in messages] == ["call-4", "call-5"]
    assert [c.arguments for c in calls] == [{"query": "q1"}, {"query": "q2"}]
    assert mock_run.call_count == 2


# ---------------------------------------------------------------------------
# run_agentic_loop tests
# ---------------------------------------------------------------------------

def test_run_agentic_loop_terminates_immediately_when_no_tool_calls():
    response = _make_terminal_response("Final answer.", "resp-1")
    client = _make_client()

    result = run_agentic_loop(
        response, client, model_defaults={}, tools=[], user=MagicMock()
    )

    assert isinstance(result, AgentResult)
    assert result.output_text == "Final answer."
    assert result.response_id == "resp-1"
    assert result.tool_calls == []
    # One turn, not zero: the response passed in is itself a billed responses.create
    # call, and accumulating at the top of the loop body is what counts it.
    assert [t.response_id for t in result.turns] == ["resp-1"]
    client.responses.create.assert_not_called()


@pytest.mark.parametrize(
    "queries",
    [
        pytest.param(["lithium"], id="one-tool-turn"),
        pytest.param(["q1", "q2"], id="two-tool-turns"),
    ],
)
def test_run_agentic_loop_continues_until_the_model_stops_calling_tools(queries):
    """The loop at one and two tool-calling turns.

    Three tests collapsed into this table — they were the same scenario at different
    turn counts, asserting one facet each (that a tool runs then the loop terminates,
    that ToolCallExecution records accumulate across iterations, that the follow-up chains
    off previous_response_id). Asserting all three at every turn count is strictly
    more coverage than the originals: continuity was previously only checked on the
    first follow-up, so a loop that re-sent resp-1 forever would have passed.
    """
    mock_run = MagicMock(return_value="doc content")
    tool = _fake_tool("search_documents", mock_run)
    user = MagicMock()

    # One tool-calling response per query, then a terminal one that ends the loop.
    tool_turns = [
        _make_tool_call_response(f"resp-{i + 1}", query=q) for i, q in enumerate(queries)
    ]
    terminal_id = f"resp-{len(queries) + 1}"
    # The first response is the one run_assistant creates and passes in; only the rest
    # come back from the client.
    client = _make_client(
        *tool_turns[1:], _make_terminal_response("Final answer.", terminal_id)
    )

    result = run_agentic_loop(
        tool_turns[0], client, model_defaults={}, tools=[tool], user=user
    )

    # The tool ran once per turn, with user bound at each dispatch.
    assert mock_run.call_args_list == [call(user=user, query=q) for q in queries]
    # ToolCallExecution records from every iteration accumulate into one flat list.
    assert [c.arguments for c in result.tool_calls] == [{"query": q} for q in queries]
    assert all(c.status is ToolCallStatus.OK for c in result.tool_calls)
    # Loop continuity: each follow-up chains off the id of the response it answers,
    # so the chain advances resp-1 -> resp-2 -> ... rather than repeating resp-1.
    assert [
        c.kwargs["previous_response_id"] for c in client.responses.create.call_args_list
    ] == [turn.id for turn in tool_turns]
    # Terminating returns the *last* response's text and id, not the first.
    assert result.output_text == "Final answer."
    assert result.response_id == terminal_id
    # One TurnUsage per responses.create, the tool-calling turns *and* the terminal one.
    # Accumulating anywhere but the top of the loop body drops one end or the other.
    assert [t.response_id for t in result.turns] == [
        *(turn.id for turn in tool_turns),
        terminal_id,
    ]
    # These responses are MagicMocks, so every usage leaf reads back as a Mock rather
    # than an int. The isinstance guard must turn that into None: MagicMock implements
    # __add__/__radd__, so without it the eval would sum mock objects into the CSV with
    # this suite green.
    assert all(t.input_tokens is None for t in result.turns)
    assert all(t.reasoning_output_tokens is None for t in result.turns)


def test_run_agentic_loop_feeds_each_turns_tool_output_back_to_the_model():
    """The tool's result actually reaches the model on the following turn.

    Kept separate from the loop test above because that test never looks at `input`.
    It asserts previous_response_id, the dispatch call args, the accumulated records
    and the final text/id — every one of which still holds if the continuation sends
    an empty or stale payload. So the single thing a tool-calling turn exists to do,
    hand the tool's output back, was the one thing unasserted: the model would answer
    from nothing while the whole suite stayed green.

    Asserted on every continuation rather than just the first, and with a different
    output per turn, because the two failure modes are distinct. The loop rebuilds
    tool_output_schemas per iteration, so a bug that re-sent turn 1's payload forever
    is not the same as one that sent none at all, and identical outputs could not tell
    them apart.
    """
    # Distinct outputs per turn so a stale payload is visible, not just a missing one.
    mock_run = MagicMock(side_effect=["first result", "second result"])
    tool = _fake_tool("search_documents", mock_run)

    tool_turns = [_make_tool_call_response("resp-1"), _make_tool_call_response("resp-2")]
    client = _make_client(
        tool_turns[1], _make_terminal_response("Final answer.", "resp-3")
    )

    run_agentic_loop(
        tool_turns[0], client, model_defaults={}, tools=[tool], user=MagicMock()
    )

    # The function_call_output payload the loop hands back on each continuation call.
    # call_id must echo the model's own call_id or the API cannot pair the output with
    # the call it answers.
    assert [c.kwargs["input"] for c in client.responses.create.call_args_list] == [
        [
            {
                "type": "function_call_output",
                "call_id": "call-loop",
                "output": "first result",
            }
        ],
        [
            {
                "type": "function_call_output",
                "call_id": "call-loop",
                "output": "second result",
            }
        ],
    ]


# ---------------------------------------------------------------------------
# _turn_usage tests
# ---------------------------------------------------------------------------

def _make_usage(input_tokens, cached_tokens, output_tokens, reasoning_tokens):
    """A real ResponseUsage, built by the SDK rather than by us.

    This is the only input in the suite not constructed from field names we chose, and
    that is the entire point. Everywhere else the mock is built from the same names the
    implementation reads, so the two agree no matter what the SDK actually calls them —
    a misspelling would return None, blank the column, and pass every test, because
    None is also the legitimate encoding of "usage was missing". Here pydantic rejects
    a name we invented.

    total_tokens is required by the model so it is supplied, and deliberately asserted
    nowhere: it is input + output, and TurnUsage does not carry it.
    """
    return ResponseUsage(
        input_tokens=input_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=cached_tokens),
        output_tokens=output_tokens,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=reasoning_tokens),
        total_tokens=input_tokens + output_tokens,
    )


def test_turn_usage_reads_every_field_off_a_real_response_usage():
    """The four SDK field names, and which TurnUsage field each one lands in.

    Every value is distinct, so a transposition — input read into output, cached into
    reasoning — fails here instead of reaching the CSV as four plausible numbers.
    Asserting merely that the fields are non-None could not catch that: a wrong mapping
    is populated too.
    """
    response = MagicMock()
    response.id = "resp-1"
    response.usage = _make_usage(
        input_tokens=100, cached_tokens=16, output_tokens=25, reasoning_tokens=9
    )

    assert _turn_usage(response) == TurnUsage(
        response_id="resp-1",
        input_tokens=100,
        cached_input_tokens=16,
        output_tokens=25,
        reasoning_output_tokens=9,
    )


def test_turn_usage_is_all_none_when_the_response_carries_no_usage():
    """Response.usage is Optional in the SDK, so this is the realistic unknown case.

    Also the traversal guard: usage.input_tokens_details would raise on a missing
    usage, before any leaf check ran. The turn is still recorded — response_id survives
    — so turn_count stays truthful even when the counts are unknown.
    """
    response = MagicMock()
    response.id = "resp-1"
    response.usage = None

    usage = _turn_usage(response)

    assert usage.response_id == "resp-1"
    assert usage.input_tokens is None
    assert usage.cached_input_tokens is None
    assert usage.output_tokens is None
    assert usage.reasoning_output_tokens is None


def test_turn_usage_reads_an_unrecognized_leaf_as_none_rather_than_raising():
    """A leaf that is present but not an int is unknown, and must not fail the request.

    _turn_usage runs on the web request path, so raising here would let telemetry break
    a user's answer. That makes silence the real hazard, which is why the isinstance
    guard exists: without it these values would be carried into the eval and summed,
    and MagicMock's __radd__ means even the arithmetic would not complain.

    SimpleNamespace rather than ResponseUsage here on purpose — pydantic would reject
    these values outright, and the shape under test is precisely the one the SDK would
    never produce.
    """
    response = SimpleNamespace(
        id="resp-1",
        usage=SimpleNamespace(
            input_tokens="100",
            input_tokens_details=SimpleNamespace(cached_tokens=True),
            output_tokens=None,
            output_tokens_details=SimpleNamespace(reasoning_tokens=9),
        ),
    )

    usage = _turn_usage(response)

    # A string that looks like a number is still not a number.
    assert usage.input_tokens is None
    # bool is an int subclass, so True would otherwise record as a count of 1.
    assert usage.cached_input_tokens is None
    assert usage.output_tokens is None
    # The one recognized leaf is still read: a single bad field does not blank the rest.
    assert usage.reasoning_output_tokens == 9
