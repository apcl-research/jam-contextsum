import google.generativeai as genai
import pickle
import tqdm
import time

contextsum_dat = pickle.load(open("/nfs/projects/EyeContext/summaries/eyecontext_openai_summaries.pkl", "rb"))
projects_dat = pickle.load(open("../gpt/all_project_code.pkl", "rb"))

key = ""
with open('gemini.key', 'r') as f:
    key = f.readline().strip()

genai.configure(api_key=key)

model = genai.GenerativeModel('gemini-pro')

## prevent output from being blocked
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]





def generate_sum(code, context):
  prompt = f'Consider the following Java method: {code}\nAnd consider the following description of Java methods that CALL that first Java method: {context}\nNow, write a one-sentence description of WHY the first method is used.  The sentence should start with "This method is used to".  The WHY description should only include information from the methods that CALL the first method and not already in the first method.  Output the sentence only, without commentary or quotes.'
  answer = model.generate_content(prompt, safety_settings=safety_settings)
  return answer.text


def generate_sum_context(context):
  prompt = f'Write a short description of each of the following Java methods, do not duplicate the code in your answer, just give a list of the descriptions in paragraph form for each description: {context}'
  answer = model.generate_content(prompt, safety_settings=safety_settings)
  return answer






newdat = list()

for i in tqdm.tqdm(range(0, len(contextsum_dat[:]))):
    wrk = contextsum_dat[i].copy()
    code = wrk['code'][0]
    project_name = wrk['project']
    project_code = projects_dat[project_name]
    project_code = ''.join(project_code)
    project_code = list(project_code)[:50000]
    project_code = ''.join(project_code)
    contextsum_vanilla = generate_sum(code,project_code)
    wrk['gemini_baseline_summary'] = contextsum_vanilla
    context = wrk['gpt_summary']
    contextsum_summary = generate_sum(code, context)
    wrk['gemini_context_summary'] = contextsum_summary
    newdat.append(wrk)
    #code = wrk['code'][0]
    #context = wrk['gpt_summary']
    #contextsumA = generate_openai_sum_context(context)
pickle.dump(newdat, open('gemini_summary.pkl', 'wb'))

