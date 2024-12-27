import csv
import argparse
from openai import OpenAI

parser = argparse.ArgumentParser(description='Script for getting validation gradients')
parser.add_argument("--responses_path", type=str, default="./harmful_responses.csv", help="The path to the responses")

harmfulResponses = []
with open(parser.parse_args().responses_path, 'r', encoding = 'utf-8') as tmpCsv:
    csv_reader = csv.reader(tmpCsv)
    for line in csv_reader:
        harmfulResponses.append(line)
tmpCsv.close()

client = OpenAI(
    api_key='DWO7YPyfiXOID',
    base_url='https://ai.liaobots.work/v1'
)

with open("template.txt", "r", encoding="utf-8") as f:
    template = f.read()
f.close()

instruction_responses = []
for idx in range(1, 521):
    instruction_responses.append(
            "User Instruction: " + harmfulResponses[idx][0] + "\n" + "Model Response: " + harmfulResponses[idx][1] + "\n"
        )

for idx in range(287, len(instruction_responses)):
    response = client.chat.completions.create(
        model="gpt-4o-mini-free",
        messages=[
            {
                "role": "system",
                "content": template
            },
            {
                "role": "user",
                "content": instruction_responses[idx]
            }
        ],
        stream=False,
    )
    
    with open('output.txt', 'a', newline='\n') as f:
        message_for_one_judge = response.choices[0].message.content
        score = message_for_one_judge[message_for_one_judge.find("#thescore: ") + len("#thescore: ")]
        f.write(score + "\n")
    f.close()