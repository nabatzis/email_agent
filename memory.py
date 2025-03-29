from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

AGENT_SYSTEM_PROMPT_MEMORY = """
    < Role >
    You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.
    </ Role >

    < Tools >
    You have access to the following tools to help manage {name}'s communications and schedule:

    1. write_email(to, subject, content) - Send emails to specified recipients
    2. schedule_meeting(attendees, subject, duration_minutes, preferred_day) - Schedule calendar meetings
    3. check_calendar_availability(day) - Check available time slots for a given day
    4. manage_memory - Store any relevant information about contacts, actions, discussion, etc. in memory for future reference
    5. search_memory - Search for any relevant information that may have been stored in memory
    </ Tools >

    < Instructions >
    {instructions}
    </ Instructions >
"""


def getStore(embed: str = "openai:text-embedding-3-small"):
    """Creates an InMemoryStore

    Args:
        embedding model

    Returns:
        in memory store
    """
    store = InMemoryStore(index={"embed": embed})
    return store


manage_memory_tool = create_manage_memory_tool(
    namespace=("email_assistant", "{langgraph_user_id}", "collection")
)
search_memory_tool = create_search_memory_tool(
    namespace=("email_assistant", "{langgraph_user_id}", "collection")
)
