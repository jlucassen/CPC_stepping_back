from llm import LLM

llm = LLM("gpt-3.5-turbo")
print(llm.chat_completion_false_start("What is latin for Ant? (A) Apoidea, (B) Rhopalocera, (C) Formicidae", "The answer is ("))