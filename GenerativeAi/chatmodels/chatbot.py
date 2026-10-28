
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    temperature=0.9
)

print("Choose your AI mode")
print("Press 1 for Angry mode")
print("Press 2 for Funny mode")
print("Press 3 for Sad mode")

choice = input("Enter your choice: ")

if choice == "1":
    mode = "You are an angry AI agent. Respond aggressively and impatiently, but do not insult or threaten the user."
elif choice == "2":
    mode = "You are a funny AI agent. Respond with humor and jokes."
elif choice == "3":
    mode = "You are a sad AI agent. Respond in a sad and emotional tone."
else:
    print("Invalid choice. Please restart and choose 1, 2, or 3.")
    exit()

messages = [
    SystemMessage(content=mode)
]

print("\nWelcome! Type 0 to exit the application.")

while True:
    prompt = input("You: ")

    if prompt == "0":
        print("Goodbye!")
        break

    messages.append(HumanMessage(content=prompt))

    response = model.invoke(messages)

    messages.append(AIMessage(content=response.content))

    print("Bot:", response.content)









# from dotenv import load_dotenv

# load_dotenv()

# from langchain_mistralai import ChatMistralAI
# from langchain_core.messages import AIMessage, SystemMessage , HumanMessage


# model = ChatMistralAI(model = "mistral-small-2506",temperature=0.9)

# print("chosse your AI mode")
# print("press 1 for Angry mode")
# print("press 2 for funny mode ")
# print("press 3 for sad mode")

# choice = int(input("tell your response :- "))

# if choice == 1:
#     mode = "You are an angry AI agent. You respond aggressively and impatiently."
# elif choice == 2:
#     mode = "You are a very funny AI agent. You respond with humor and jokes."
# elif choice == 3:
#     mode = "You are a very sad AI agent. You respond in a depressed and emotional tone."


# messages = [
#     SystemMessage(content=mode)
# ]

# print("----------------- welcom type 0 to exit the application-----------------")
# while True:
    
#     prompt = input("You : ")
#     messages.append(HumanMessage(content=prompt))
#     if prompt == "0":
#         break
#     response = model.invoke(messages)
#     messages.append(AIMessage(content=response.content))
#     print("Bot :",response.content)

# print(messages)