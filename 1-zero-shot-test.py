from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from utils import print_llm_result
from datetime import datetime 
import os
import platform

load_dotenv(find_dotenv())

#Customized Romilson Lemes
#--------------------------------------------
print(f"Operational System: {platform.system()} - Os Name: {os.name}")
#Clear Screen
# os.system('cls' if os.name == 'nt' else 'clear')

# exit() # Exit System

# Start Prompt Engineering Implementation
if (load_dotenv(find_dotenv()) == True):
    print("OPENAI_API_KEY was Found in the same folder..")
#--------------------------------------------
startedTime = datetime.now() 
startedTimeFormat = startedTime.strftime("%d/%m/%Y %H:%M:%S")  
print(f"Started this Process: {startedTimeFormat}")

msg1 = "Give the result about this expression (((45*9) /3)) * 1898) / 1.85"

# msg2 = """
# Find the user intent in the following text: 
# I'm looking for a Brazilian restaurant around Calgary who has a good rating for Brazilian food.
# """

# msg2 = "What is the best  Brazilian Steakhouse restaurant around Calgary who has a good rating for Brazilian barbecue."

# msg3 = "What's Canada's capital? Respond only with the city name."


#MODEL LIST:
#===================
usedModel = "gpt-4.1"

"""
gpt-5
gpt-5-mini
o1
o3
gpt-4.1
gpt-3.5-turbo

"""
print(f"We are using the model: [ {usedModel} ] for this research")

llm = ChatOpenAI(model=f"{usedModel}")
response1 = llm.invoke(msg1)
# response2 = llm.invoke(msg2)
# response3 = llm.invoke(msg3)

print(f"\nQuestion 1: {msg1} \n")
print_llm_result(msg1, response1)
# print(f"\nQuestion 2: {msg2} \n")
# print_llm_result(msg2, response2)
# print(f"\nQuestion 3: {msg3} \n ")
# print_llm_result(msg3, response3)

finishedTime = datetime.now() 
finishedTimeFormat = startedTime.strftime("%d/%m/%Y %H:%M:%S") 
print(f"Finished this Process: {finishedTime}")
