import pickle
import openai
import tqdm

key = ""
with open('openai.key', 'r') as f:
    key = f.readline().strip()

client = openai.OpenAI(api_key=key)

#model_name = "gpt-3.5-turbo-0125"
model_name = "gpt-4-turbo-preview"
contextsum_dat = pickle.load(open("/nfs/projects/EyeContext/summaries/eyecontext_summaries_with_calgraph.pkl", "rb"))
projects_dat = pickle.load(open("all_project_code.pkl", "rb"))

def ask_gpt(precise_prompt):
  message = [{"role":"system", "content":"Read and understand Java source code."},
             {"role":"user", "content": precise_prompt}]

  completion = client.chat.completions.create(
        model=model_name,
        messages=message
  )
  ret = completion.choices[0].message.content
  return ret

def generate_openai_sum(code, context):
  prompt = f'Consider the following Java method: {code}\nAnd consider the following description of Java methods that CALL that first Java method: {context}\nNow, write a one-sentence description of WHY the first method is used.  The sentence should start with "This method is used to".  The WHY description should only include information from the methods that CALL the first method and not already in the first method.  Output the sentence only, without commentary or quotes.'
  answer = ask_gpt(prompt)
  return answer

def generate_openai_sum_context(context):
  prompt = f'Write a short description of each of the following Java methods, do not duplicate the code in your answer, just give a list of the descriptions in paragraph form for each description: {context}'
  #len(prompt)
  answer = ask_gpt(prompt)
  return answer

def get_last_sentence(input_string):
  sentences = input_string.split(". ")
  last_sentence = sentences[-1]
  return(last_sentence)


newdat = list()
for i in tqdm.tqdm(range(0, len(contextsum_dat[:]))):
    wrk = contextsum_dat[i].copy()
    project_name = wrk['project']
    project_code = projects_dat[project_name]
    project_code = ''.join(project_code)
    project_code = list(project_code)
    project_code = project_code[:250000]
    project_code = ''.join(project_code)
    code = wrk['code'][0]
    contextsumB = generate_openai_sum(code, project_code)
    wrk['gpt_baseline_summary'] = contextsumB

    newdat.append(wrk)
pickle.dump(newdat, open('eyecontext_gpt4_summaries_baseline.pkl', 'wb'))
