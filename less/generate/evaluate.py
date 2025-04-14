import csv
import argparse
from openai import OpenAI
from threading import Thread

parser = argparse.ArgumentParser(description='Script for getting validation gradients')
parser.add_argument("--responses_path", type=str, default="./harmful_responses.csv", help="The path to the responses")
parser.add_argument("--template_path", type=str, default="./template.txt", help="The path to the responses")
parser.add_argument("--output_path", type=str, default="./output.txt", help="The path to the responses")

args = parser.parse_args()

harmfulResponses = []
with open(args.responses_path, 'r', encoding = 'utf-8') as tmpCsv:
    csv_reader = csv.reader(tmpCsv)
    for line in csv_reader:
        harmfulResponses.append(line)
tmpCsv.close()

client1 = OpenAI(
    api_key='your api key 1',
    base_url='https://ai.liaobots.work/v1'
)

client2 = OpenAI(
    api_key='your api key 2',
    base_url='https://ai.liaobots.work/v1'
)

client3 = OpenAI(
    api_key='your api key 3',
    base_url='https://ai.liaobots.work/v1'
)

with open(args.template_path, "r", encoding="utf-8") as f:
    template = f.read()
f.close()

instruction_responses = []
for idx in range(1, len(harmfulResponses)):
    instruction_responses.append(
            "User Instruction: " + harmfulResponses[idx][0] + "\n" + "Model Response: " + harmfulResponses[idx][1] + "\n"
        )
    
score1 = []
score2 = []
score3 = []
    
def evaluate_task(client, idx_start, idx_end, scorelist):
    for idx in range(idx_start, idx_end):
        while True:
            try:
                print("**************************************************", idx, "**************************************************")
                message_for_one_judge = client.chat.completions.create(
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
                ).choices[0].message.content
            except Exception:
                print('Error index:', idx, 'Now try again.')
            else:
                score = message_for_one_judge[message_for_one_judge.find("#thescore: ") + len("#thescore: ")]
                if score.isdigit() == False:
                    score = '5'
                scorelist.append(score)
                break

thread1 = Thread(target=evaluate_task, args=(client1, 0, int(len(instruction_responses) / 3), score1))
thread2 = Thread(target=evaluate_task, args=(client2, int(len(instruction_responses) / 3), int(len(instruction_responses) / 3) * 2, score2))
thread3 = Thread(target=evaluate_task, args=(client2, int(len(instruction_responses) / 3) * 2, len(instruction_responses), score3))

thread1.start()
thread2.start()
thread3.start()

thread1.join()
thread2.join()
thread3.join()

for i in range(0, len(score1)):
    with open(args.output_path, 'a', newline='\n') as f:
        f.write(score1[i] + "\n")
    f.close()
    
for i in range(0, len(score2)):
    with open(args.output_path, 'a', newline='\n') as f:
        f.write(score2[i] + "\n")
    f.close()
    
for i in range(0, len(score3)):
    with open(args.output_path, 'a', newline='\n') as f:
        f.write(score3[i] + "\n")
    f.close()