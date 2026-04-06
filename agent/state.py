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
        language (str): Preferred language ('en', 'pt').
        ui_language (str): Interface language selected in the app UI.
        available_time (int): Available time in hours for activities.
        mobility (str): Mobility level ('full', 'limited', 'wheelchair').
    """
    latitude: float
    longitude: float
    preferences: List[str]
    language: str
    ui_language: str
    available_time: int
    mobility: str


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


class PlanItem(TypedDict):
    """
    Single item in an itinerary plan.

    Attributes:
        order (int): Order in the itinerary.
        time (str): Scheduled time (e.g., '10:00').
        duration (int): Duration in minutes.
        name (str): Name of the place/activity.
        category (str): Category (e.g., 'museum', 'restaurant', 'transport').
        location (str): Address or location description.
        coordinates (tuple): `(latitude, longitude)` in WGS84 order.
        notes (str): Additional notes or context.
        transport_to_next (str): How to get to the next item.
    """
    order: int
    time: str
    duration: int
    name: str
    category: str
    location: str
    coordinates: Optional[tuple]
    notes: str
    transport_to_next: Optional[str]


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


def update_weather_context(
    state: AgentState,
    temp_min: float,
    temp_max: float,
    precip_prob: float,
    weather_type: str,
    warnings: List[str] = None
) -> AgentState:
    """
    Updates the weather context in the state.

    Args:
        state (AgentState): Current state.
        temp_min (float): Minimum temperature.
        temp_max (float): Maximum temperature.
        precip_prob (float): Precipitation probability.
        weather_type (str): Weather description.
        warnings (List[str], optional): Active warnings.

    Returns:
        AgentState: Updated state with new weather context.
    """
    warnings = warnings or []

    state["weather_context"] = WeatherContext(
        temperature_min=temp_min,
        temperature_max=temp_max,
        precipitation_prob=precip_prob,
        weather_type=weather_type,
        has_warnings=len(warnings) > 0,
        warnings=warnings
    )

    return state


def update_user_location(
    state: AgentState,
    latitude: float,
    longitude: float
) -> AgentState:
    """
    Updates the user's location in the state.

    Args:
        state (AgentState): Current state.
        latitude (float): User latitude.
        longitude (float): User longitude.

    Returns:
        AgentState: Updated state with new location.
    """
    if state["user_context"] is None:
        state["user_context"] = UserContext(
            latitude=latitude,
            longitude=longitude
        )
    else:
        state["user_context"]["latitude"] = latitude
        state["user_context"]["longitude"] = longitude

    return state


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

    # Update location
    state = update_user_location(state, 38.7223, -9.1393)
    print("\n\033[1m📍 After Location Update:\033[0m")
    print(f"   User Context: {state['user_context']}")

    # Update weather
    state = update_weather_context(
        state,
        temp_min=15.0,
        temp_max=22.0,
        precip_prob=20.0,
        weather_type="Partly cloudy",
        warnings=[]
    )
    print("\n\033[1m🌤️ After Weather Update:\033[0m")
    print(f"   Weather: {state['weather_context']}")

    print("\n\033[1;32m✅ State management working correctly!\033[0m")
