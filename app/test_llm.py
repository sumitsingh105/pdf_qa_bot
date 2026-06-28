# app/test_llm.py

from llm import generate_answer

answer = generate_answer(
    "What is Python?",
    "Python is a programming language."
)

print(answer)