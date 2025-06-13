import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq
import json
import re
#---------------------------------------------------------------------------------------------------------------------

Groq_api_key = "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"

client =Groq(api_key=Groq_api_key)

#--------------------------------------------------------------------------------------------------------------
#Load and and preprocess the dataset

def load_data_set():
    df = pd.read_csv('dataset/household_power_consumption.txt',
             	sep=';',
             	parse_dates={'datetime': ['Date', 'Time']},
             	infer_datetime_format=True,
             	na_values=['?'],
             	low_memory=False)

    df = df.dropna()
    df['Global_active_power'] = df['Global_active_power'].astype(float)
    df = df.set_index('datetime')
    return df

#--------------------------------------------------------------------------------------------------------------
# Function to generate pandas code from natural language question using GROQ API

# Function to generate pandas code from natural language question using GROQ API

def generate_pandas_code(natural_language_question, model_name, df_info):
    df_info_str = json.dumps(df_info)

    prompt = f"""You are a python expert and you are given a dataset which is preprocessed using below code:

    df = pd.read_csv('dataset/household_power_consumption.txt',
                     sep=';',
                     parse_dates={{'datetime': ['Date', 'Time']}},
                     infer_datetime_format=True,
                     na_values=['?'],
                     low_memory=False)

    df = df.dropna()
    df['Global_active_power'] = df['Global_active_power'].astype(float)
    df = df.set_index('datetime')

    and following are the column names in the dataset after preprocessing where as you can see that the index is set to datetime:
    {df_info_str}


    Provide me with only python pandas code for below natural language questions no theory, explanation is needed. Please only provide the pandas code with necessary imports which should print the result when executed:
    {natural_language_question}

    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model_name,
            temperature=0.1,
        )
        return chat_completion.choices[0].message.content

    except Exception as e:
        print(f"GROQ API error: {e}")
        return None
#--------------------------------------------------------------------------------------------------------------
# Function to execute the generated pandas code and display the output

def execute_query(query,model_name,df):
    df_info = {'columns': df.columns.tolist()}

    code = generate_pandas_code(query,model_name,df_info)
    #print("Python panda code is: \n",code)

    # Clean the code by removing markdown formatting so that it can be executed
    if code:
        code_clean = re.sub(r"```(?:python)?\n?|```", "", code).strip()
        code_clean = re.sub(r"<think>.*?</think>", "", code_clean, flags=re.DOTALL).strip()
        code_clean = code_clean.strip().split('\n')
        valid_lines = []
        for line in code_clean:
            # Include lines that look like real Python code (simple heuristic)
            if line.strip().startswith(("print", "df","pd","plt", "#", "import", "average", "result")) or "=" in line:
                valid_lines.append(line)

        final_code = '\n'.join(valid_lines)
        fixed_code = re.sub(
            r"df\[(df\.index\.\w+\s*==\s*\d+)\]\[(df\.index\.\w+\s*==\s*\d+)\]",
            r"df[(\1) & (\2)]",
            final_code
        )
        print("------------------------------------------------------------------------\nPython pandas Code is:\n\n", fixed_code)

    try:
        print("------------------------------------------------------------------------\nOutput of the code is:\n")
        exec(fixed_code, {'df': df}) #execute the code in the context of the dataframe
    except Exception as e:
        print(f"Error executing code: {e}")
        return False
#--------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading dataset...")
    df = load_data_set()
    
    if df is not None:
        print("Dataset loaded successfully./n")

        while True:
            print("Select any one LLM model you want to use for query execution:")
            print("1. llama-3.3-70b-Versatile\n2. qwen/qwen3-32b\n3. deepseek-r1-distill-llama-70b\n4. gemma2-9b-it\n5. mistral-saba-24b")
            model_name = input("Enter the name of the model:")
            if model_name not in ["llama-3.3-70b-Versatile", "qwen/qwen3-32b", "deepseek-r1-distill-llama-70b", "gemma2-9b-it", "mistral-saba-24b"]:
                print("Invalid model name. Please try again.")
                continue

            query = input("Enter your natural language question (or type 'exit' to quit): ")
            if query.lower() == 'exit':
                quit()
            execute_query(query,"llama-3.3-70b-Versatile",df)
            print("\nQuery executed successfully.\n------------------------------------------------------------------------\n")
    else:
        print("Failed to load dataset. Please check the file path and format.")
#--------------------------------------------------------------------------------------------------------------cls