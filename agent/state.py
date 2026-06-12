# ==========================================================================
# Master Thesis - Agent State Definition
#   - André Filipe Gomes Silvestre, 20240502
#
#   Defines the state schema for the LangGraph agent.
#   Uses TypedDict for type safety and clear state management.
# ==========================================================================

from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class UserContext(TypedDict, total=False):
    """
    User context information for personalized responses.

    Attributes:
        latitude (float): User's current latitude.
        longitude (float): User's current longitude.
        preferences (List[str]): User interests (e.g., 'museums', 'food', 'nature').
        language (str): Effective response language ('en', 'pt').
        ui_language (str): Interface language selected in the app UI.
        detected_language (str): ISO-639-1 code (or extended code) of the
            language detected from the latest user query, e.g. ``fr``, ``de``,
            ``ja``. Populated by :func:`agent.utils.response_formatter.resolve_output_language`.
        requires_bilingual_note (bool): True when the user wrote in a language
            other than PT or EN and the assistant should surface a bilingual
            note in the final response explaining it is optimized for PT/EN.
        available_time (int): Available time in hours for activities.
        mobility (str): Mobility level ('full', 'limited', 'wheelchair').
    """
    latitude: float
    longitude: float
    preferences: List[str]
    language: str
    ui_language: str
    detected_language: str
    requires_bilingual_note: bool
    available_time: int
    mobility: str
    conversation_anchors: dict


class WeatherContext(TypedDict, total=False):
    """
    Weather context from IPMA API.

    Attributes:
        temperature_min (float): Minimum temperature in Celsius.
        temperature_max (float): Maximum temperature in Celsius.
        precipitation_prob (float): Probability of precipitation (0-100).
        weather_type (str): Weather description.
        has_warnings (bool): Whether there are active weather warnings.
        warnings (List[str]): List of active warning descriptions.
    """
    temperature_min: float
    temperature_max: float
    precipitation_prob: float
    weather_type: str
    has_warnings: bool
    warnings: List[str]


class TransportContext(TypedDict, total=False):
    """
    Transport status context.

    Attributes:
        metro_status (dict): Status of each metro line.
        carris_alerts (int): Number of active bus alerts.
        train_delays (int): Number of delayed trains.
        last_updated (str): ISO timestamp of last update.
    """
    metro_status: dict
    carris_alerts: int
    train_delays: int
    last_updated: str


class AgentState(TypedDict):
    """
    Main state schema for the Lisbon Urban Assistant agent.

    This state is passed through the LangGraph workflow and contains
    all context needed for personalized responses.

    Attributes:
        messages (List[BaseMessage]): Conversation history with message reducer.
        user_context (UserContext): User-specific information.
        weather_context (WeatherContext): Current weather data.
        transport_context (TransportContext): Transport status data.
        current_plan (List[dict]): Current itinerary items.
        session_id (str): Unique session identifier.
        last_tool_result (str): Result from the last tool call.

        Multi-Agent Orchestration:
        next_agent (str): Name of the next agent to execute, when applicable.
        agents_to_call (List[str]): Queue or routing decision from the supervisor.
        candidate_pois (List[dict]): Candidate places from retrieval.
        events_data (List[dict]): Candidate event records from retrieval.
        agent_outputs (dict): Collected outputs from specialized agents.
        iteration_count (int): Loop-prevention and execution-tracking counter.
    """
    # Conversation history (uses add_messages reducer for proper appending)
    messages: Annotated[List[BaseMessage], add_messages]

    # Context information
    user_context: Optional[UserContext]
    weather_context: Optional[WeatherContext]
    transport_context: Optional[TransportContext]

    # Planning state
    current_plan: Optional[List[dict]]

    # Session metadata
    session_id: Optional[str]
    last_tool_result: Optional[str]

    # Multi-Agent orchestration fields
    next_agent: Optional[str]              # Current agent being executed, if tracked
    agents_to_call: Optional[List[str]]    # Supervisor routing decision
    candidate_pois: Optional[List[dict]]   # Retrieved places/attractions considered for planning
    events_data: Optional[List[dict]]      # Retrieved event records considered for planning
    agent_outputs: Optional[dict]          # Outputs from each agent {agent_name: output}
    iteration_count: Optional[int]         # Execution-tracking metadata for loop prevention


def create_initial_state(session_id: str = None) -> AgentState:
    """
    Creates an initial empty state for a new conversation.

    Args:
        session_id (str, optional): Session identifier.
                                   Generated if not provided.

    Returns:
        AgentState: Initial state with empty values.
    """
    from uuid import uuid4

    return AgentState(
        messages=[],
        user_context=None,
        weather_context=None,
        transport_context=None,
        current_plan=None,
        session_id=session_id or str(uuid4())[:8],
        last_tool_result=None,
        # Multi-Agent fields
        next_agent=None,
        agents_to_call=None,
        candidate_pois=None,
        events_data=None,
        agent_outputs={},
        iteration_count=0
    )


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 Agent State Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")

    # Create initial state
    state = create_initial_state()
    print("\n\033[1m📋 Initial State:\033[0m")
    print(f"   Session ID: {state['session_id']}")
    print(f"   Messages: {len(state['messages'])}")
    print(f"   User Context: {state['user_context']}")

    print("\n\033[1;32m✅ State management working correctly!\033[0m")
