from langchain.tools import tool


@tool #decorator for creating tool 
def get_greeting(name : str) -> str: #type hints : -> use pointer to indicate the return type of the function
    """Generate a greeting message for a user""" #docstring = to tell what the function does

    return f"Hello {name}, Welcome to the AI world" 


result = get_greeting.invoke({"name":"Shubham"}) #invoke the tool with the input parameter
print(result)

print(get_greeting.name)
print(get_greeting.description)
print(get_greeting.args)
