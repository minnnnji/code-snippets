import lmstudio as lms

model = lms.llm("hyperclovax-seed-text-instruct-0.5b")
result = model.respond("What is the meaning of life? 에 대한 답변을 한글로 설명해줘")

print(result)