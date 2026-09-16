import json
import logging

from api.views.assistant.assistant_types import (
    AgentResult,
    ToolCallExecution,
    ToolCallStatus,
    TurnUsage,
)

logger = logging.getLogger(__name__)


def run_agentic_loop(
    response, client, model_defaults: dict, tools: list, user
) -> AgentResult:

    # Every tool call the agentic loop made before exiting
    agentic_loop_tool_call_executions= []
    # Token usage for every responses.create call, one entry per iteration
    agentic_loop_turns: list[TurnUsage] = []

    while True:
        # At the top of the body, so the initial response and the terminal turn are each counted exactly once

        # _turn_usage never raises: it runs on the web request path, so an unrecognized usage shape must not fail a user's request.
        agentic_loop_turns.append(_turn_usage(response))

        # user is threaded through so tools that need it get it at dispatch time
        tool_output_schemas, tool_call_executions = handle_tool_calls(response, tools, user)

        # TODO: Decide whether to add turn: int to ToolCallExecution — without it, the flat tool_calls can't be split back into turns
        # .extend splices every iteration's list of tools into one list
        agentic_loop_tool_call_executions.extend(tool_call_executions)

        # Exit agentic loop when model response doesn't contain any tool calls
        if not tool_output_schemas:
            return AgentResult(
                output_text=response.output_text,
                response_id=response.id,
                tool_calls=agentic_loop_tool_call_executions,
                turns=agentic_loop_turns,
            )

        #TODO: Add error handling to collect partial AgentResult tool calls
        response = client.responses.create(
            input=tool_output_schemas,
            previous_response_id=response.id,
            **model_defaults,
        )


def _int_or_none(obj, field: str) -> int | None:
    """Read one integer leaf, or None when it is missing or not an int.

    bool is excluded deliberately: it is an int subclass, so True would otherwise be
    recorded as a token count of 1.
    """
    value = getattr(obj, field, None)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _turn_usage(response) -> TurnUsage:
    """Token usage for one response, never raising.

    This runs on the web request path as well as in the eval, so an unrecognized usage
    shape must not fail a user's request.

    getattr's default guards the *traversal*, not only the leaves: when usage is
    missing, usage.output_tokens_details would raise before any leaf check ran.
    getattr(None, ...) returns None instead, collapsing the whole chain.

    The isinstance guard is what lets the tests fail. MagicMock implements
    __add__/__radd__, so a mocked usage would otherwise accumulate into the CSV as mock
    objects with the suite green.
    """

    #  getattr's default guards the traversal rather than only the leaves,
    # since usage.output_tokens_details would raise before any leaf check when usage is missing

    # That guard makes silent blanks the hazard, so the field names are pinned by a
    # test building a real ResponseUsage - the only input in the suite not
    # constructed from names we chose, and so the only one where a misspelling can
    # fail rather than quietly blanking a column. Verified against openai 2.29.0.
    
    usage = getattr(response, "usage", None)
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)

    return TurnUsage(
        response_id=response.id,
        input_tokens=_int_or_none(usage, "input_tokens"),
        cached_input_tokens=_int_or_none(input_details, "cached_tokens"),
        output_tokens=_int_or_none(usage, "output_tokens"),
        reasoning_output_tokens=_int_or_none(output_details, "reasoning_tokens"),
    )


def handle_tool_calls(
    response, tools: list, user
) -> tuple[list[dict], list[ToolCallExecution]]:

    # Index the tools by name so a model-supplied call name can be looked up. .get()
    # returns None for an unknown name, handled explicitly below.
    tools_by_name = {tool.name: tool for tool in tools}

    tool_output_schemas = []
    tool_call_executions: list[ToolCallExecution] = []

    for response_item in response.output:
        if response_item.type == "reasoning":
            #logger.info(f"Reasoning step: {response_item.summary}")
            pass

        elif response_item.type == "function_call":

            tool_output, tool_call_execution = _execute_function_call(response_item, tools_by_name, user)

            tool_output_schemas.append(
                {
                    "type": "function_call_output",
                    "call_id": response_item.call_id,
                    "output": tool_output,
                }
            )
            
            tool_call_executions.append(tool_call_execution)


    return tool_output_schemas, tool_call_executions


def _execute_function_call(
    response_item, tools_by_name: dict, user
) -> tuple[str, ToolCallExecution]:

    target_tool = tools_by_name.get(response_item.name)
    
    # Parsed below; stays None if the model's argument JSON can't be parsed,
    # so a FAILED record still reports whatever we managed to read.
    arguments = None

    if target_tool is None:
        msg = f"ERROR - No tool registered for function call: {response_item.name}"
        logger.error(msg)
        return msg, ToolCallExecution(
            name=response_item.name,
            status=ToolCallStatus.UNREGISTERED,
            error=msg,
        )

    try:
        arguments = json.loads(response_item.arguments)
        logger.info(
            f"Invoking tool: {response_item.name} with arguments: {arguments}"
        )
        tool_output = target_tool.run(user=user, **arguments)
        logger.info(f"Tool {response_item.name} completed successfully")
        return tool_output, ToolCallExecution(
            name=response_item.name,
            status=ToolCallStatus.OK,
            arguments=arguments,
            output=tool_output,
        )
    except Exception as e:
        msg = f"Error executing function call: {response_item.name}: {e}"
        logger.error(msg, exc_info=True)
        return msg, ToolCallExecution(
            name=response_item.name,
            status=ToolCallStatus.FAILED,
            arguments=arguments,
            error=str(e),
        )
