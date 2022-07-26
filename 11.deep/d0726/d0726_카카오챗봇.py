import pandas as pd 

path = "11.deep/d0726/"
chatbotData=pd.read_csv(path+"ChatBotData.csv")
question, answer = list(chatbotData["Q"]) , list(chatbotData["A"])
print(len(question))
print(len(answer))

for i in range(10):
    print("질문:" + question[i])
    print("답변:" + answer[i])
    print(" ")