#written by Diego Ford 
#!pip install pandas as pd

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
#load in model
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")



file = open("prompt 2.txt", "r")

prompt = file.read()

print(prompt)



inputs = tokenizer(prompt + " RT @HowleyReporter Trump has opportunity with audits, recounts, Supreme Court (look at Dominion Voting Systems)If it doesn't happen, go full legislative fight in the Electoral College. This election was a disgrace -- starting with Fauci's first press conference.RULE OF LAW. Constitution counts", return_tensors="pt")

outputs = model.generate(**inputs)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))



complete_df = pd.read_csv("combined.csv")





#fill NA values with a blank string

complete_df['Full Text'] = complete_df['Full Text'].fillna('')



def stance_detection(data):

    data["stance"] = ""

    for index,text in data.iterrows():

        inputs = tokenizer(prompt + text['Full Text'], return_tensors="pt")

        outputs = model.generate(**inputs)

        data.at[index, 'stance'] = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        print(index)

        print(text["Full Text"])

        print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    return(data)





result = stance_detection(complete_df[0:10])

result.to_csv(r"complete_stance.csv")