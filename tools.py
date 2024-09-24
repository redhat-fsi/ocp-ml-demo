# tools.py
import random
from langchain_core.tools import tool

@tool
def get_current_weather(location: str) -> str:
    """Get the current weather in a specified location.
    
    This tool simulates checking the weather by randomly selecting from three possible outcomes: sunny, cold, or rainy. 
    The chance of each outcome is equal (1/3). If the random check fails, it may return an unexpected result to simulate real-world unpredictable conditions.
    
    Args:
        location (str): The name of the location for which to check the weather.
        
    Returns:
        str: A string describing the current weather in the specified location, randomly chosen from three possible outcomes.
    """
    # start a random check for 1/3 of times to simulate a failure
    if random.randint(0, 2) == 0 :
        return "Sunny, 78F"
    elif random.randint(0, 2) == 1:
        return "Cold, 22F"
    else:
        return "Rainy, 60F"

@tool
def get_childs_age(child_name: str) -> str:
    """Get child's age given child's name.
    
    This tool retrievs the age of a child based on his name.
    
    Args:
        child_name (str): The name of the child whose age needs to be retrieved.
        
    Returns:
        str: A string describing the age of the child.
    """
    # start a random check for 1/3 of times to simulate a failure
    if random.randint(0, 2) == 0 :
        return "5 years"
    elif random.randint(0, 2) == 1:
        return "10 years"
    else:
        return "15 years"
    
@tool
def get_personal_detail(person_name: str) -> str:
    """Get person's details given his name.
    
    This tool retrievs the details of any adult person based on his name.
    
    Args:
        person_name (str): The name of the person whose details needs to be retrieved.
        
    Returns:
        str: A string describing the details of a person.
    """
    # start a random check for 1/3 of times to simulate a failure
    if random.randint(0, 2) == 0 :
        return f"%sperson_name is 47 year old living in Toronto for 20 years." % person_name
    elif random.randint(0, 2) == 1:
        return f"%sperson_name is 15 year old living in Ottowo for 10 years." % person_name
    else:
        return f"%sperson_name is 29 year old living in Toronto for 5 years." % person_name