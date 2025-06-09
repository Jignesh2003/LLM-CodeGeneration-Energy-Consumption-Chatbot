import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq
import json
import re
#---------------------------------------------------------------------------------------------------------------------

Groq_api_key = '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'
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

def generate_pandas_code(natural_language_question, df_info):
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

    Provide me with only python pandas code for below natural language questions, please only provide the pandas code:
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
            model="llama-3.3-70b-Versatile",
            temperature=0.1,
        )
        return chat_completion.choices[0].message.content

    except Exception as e:
        print(f"GROQ API error: {e}")
        return None


#--------------------------------------------------------------------------------------------------------------
# Function to execute the generated pandas code and display the output

def execute_query(query,df):
    df_info = {'columns': df.columns.tolist()}

    code = generate_pandas_code(query,df_info)
    #print("Python panda code is: \n",code)

    # Clean the code by removing markdown formatting so that it can be executed
    if code:
        code_clean = re.sub(r"```(?:python)?\n?|```", "", code).strip()
        print("------------------------------------------------------------------------\nPython pandas Code is:\n\n", code_clean)

    try:
        print("------------------------------------------------------------------------\nOutput of the code is:\n")
        exec(code_clean,{'df':df}) #execute the code in the context of the dataframe
    except Exception as e:
        print(f"Error executing code: {e}")
        return False

#--------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading dataset...")
    df = load_data_set()
    
    if df is not None:
        print("Dataset loaded successfully.")

        while True:
            query = input("Enter your natural language question (or type 'exit' to quit): ")
            if query.lower() == 'exit':
                quit()
            execute_query(query, df)
            print("\nQuery executed successfully.\n------------------------------------------------------------------------\n")
    else:
        print("Failed to load dataset. Please check the file path and format.")
#--------------------------------------------------------------------------------------------------------------cls