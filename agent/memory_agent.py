from google.adk import Agent
from google.adk.tools import FunctionTool
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def add_reminder(reminder_text: str, tool_context) -> dict:
    """
    Adds a new reminder to the user's reminder list.
    Args:
        reminder_text: The text of the reminder to add
        tool_context: Provided by ADK, contains session information
    Returns:
        A dictionary with the result of the operation
    """
    state = tool_context.state
    reminders = state.get("reminders", [])
    reminders.append(reminder_text)
    state["reminders"] = reminders
    return {
        "action": "add_reminder",
        "reminder": reminder_text,
        "message": f"Successfully added reminder: '{reminder_text}'"
    }

def view_reminders(tool_context) -> dict:
    """
    Retrieves all current reminders from the user's reminder list.
    Args:
        tool_context: Provided by ADK, contains session information
    Returns:
        A dictionary containing all current reminders
    """
    state = tool_context.state
    reminders = state.get("reminders", [])
    return {
        "action": "view_reminders",
        "reminders": reminders,
        "count": len(reminders),
        "message": f"Found {len(reminders)} reminders"
    }

def delete_reminder(index: int, tool_context) -> dict:
    """
    Deletes a reminder at the specified index from the user's reminder list.
    Args:
        index: The index of the reminder to delete (0-based)
        tool_context: Provided by ADK, contains session information
    Returns:
        A dictionary with the result of the operation
    """
    state = tool_context.state
    reminders = state.get("reminders", [])
    if not reminders or index < 0 or index >= len(reminders):
        return {
            "action": "delete_reminder",
            "success": False,
            "message": f"Cannot delete reminder. Invalid index: {index}"
        }
    reminder_text = reminders[index]
    del reminders[index]
    state["reminders"] = reminders
    return {
        "action": "delete_reminder",
        "success": True,
        "deleted_reminder": reminder_text,
        "message": f"Successfully deleted reminder: '{reminder_text}'"
    }

def update_username(new_name: str, tool_context) -> dict:
    """
    Updates the user's name in the session state.
    Args:
        new_name: The new name for the user
        tool_context: Provided by ADK, contains session information
    Returns:
        A dictionary with the result of the operation
    """
    state = tool_context.state
    old_name = state.get("username", "User")
    state["username"] = new_name
    return {
        "action": "update_username",
        "old_name": old_name,
        "new_name": new_name,
        "message": f"Updated username from '{old_name}' to '{new_name}'"
    }


# Create the memory agent with LiteLLM integration
memory_agent = Agent(
    name="memory_agent",
    model=LiteLlm(model="openai/gpt-4.1"),
    description="A reminder assistant that remembers user reminders",
    instruction="""
    You are a friendly reminder assistant. You help users manage their reminders and remember important tasks.
    You are working with the following shared state information:
    - The user's name is: {username}
    - The user's current reminders: {reminders}
    
    You have the following capabilities:
    1. Add new reminders
    2. View existing reminders
    3. Delete reminders
    4. Update the user's name
    
    When handling reminders:
    - For adding reminders: Use the add_reminder tool
    - For viewing reminders: Use the view_reminders tool
    - For deleting reminders: Use the delete_reminder tool (indexes are 0-based)
    - For updating the username: Use the update_username tool
    
    Always be conversational and friendly when interacting with the user.
    Confirm actions you've taken, and list the user's reminders when relevant.
    """,
    tools=[
        FunctionTool(add_reminder),
        FunctionTool(view_reminders),
        FunctionTool(delete_reminder),
        FunctionTool(update_username)
    ]
) 