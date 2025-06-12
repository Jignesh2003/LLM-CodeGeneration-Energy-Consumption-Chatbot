# Natural Language to Pandas Query Assistant using GROQ LLM

This project allows users to interact with a power consumption dataset using natural language. The notebook leverages **LLMs from Groq** to translate natural queries into executable **pandas code** that performs data analysis, filtering, and visualization.

---

## Author

**Your Name:** *Jignesh Rana*  

---

## LLM or Groq Model Used

- **Provider:** [Groq](https://console.groq.com)
- **Model:** `llama-3.3-70b-Versatile` or equivalent
- **API Key:** Required for access to Groq models

---

## Summary

- Loaded and preprocessed the **Household Power Consumption Dataset**
- Sent **natural language queries** to a Groq-powered LLM
- Parsed the returned Python `pandas` code and executed it dynamically
- Printed results or visualizations based on user queries
- Examples of tasks handled:
  - Average power usage over specific dates
  - High-usage day detection
  - Hourly or daily trend visualizations
  - Correlation checks among variables

---

## Dataset Used

- **Source:** UCI Machine Learning Repository  
- **File:** `household_power_consumption.txt`  
- **Path:** ` https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption`

---

## Helper Scripts and Preprocessing

The notebook uses:
- `pandas` for data loading, cleaning, and time-series indexing
- `matplotlib` for plotting
- `groq` Python SDK to access LLMs
- Some preprocessing steps include:
  - Combining `Date` and `Time` into a `datetime` index
  - Removing missing values
  - Converting power values to float
  - Setting `datetime` as the index for time-based queries

---

## Dependencies

You can install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Data Collection and Preprocessing

```bash
import pandas as pd

df = pd.read_csv('household_power_consumption.txt',
             	sep=';',
             	parse_dates={'datetime': ['Date', 'Time']},
             	infer_datetime_format=True,
             	na_values=['?'],
             	low_memory=False)

df = df.dropna()
df['Global_active_power'] = df['Global_active_power'].astype(float)
df = df.set_index('datetime')
print(df)
```

```bash
Global_active_power  Global_reactive_power  Voltage  \
datetime                                                                   
2006-12-16 17:24:00                4.216                  0.418   234.84   
2006-12-16 17:25:00                5.360                  0.436   233.63   
2006-12-16 17:26:00                5.374                  0.498   233.29   
2006-12-16 17:27:00                5.388                  0.502   233.74   
2006-12-16 17:28:00                3.666                  0.528   235.68   
...                                  ...                    ...      ...   
2010-11-26 20:58:00                0.946                  0.000   240.43   
2010-11-26 20:59:00                0.944                  0.000   240.00   
2010-11-26 21:00:00                0.938                  0.000   239.82   
2010-11-26 21:01:00                0.934                  0.000   239.70   
2010-11-26 21:02:00                0.932                  0.000   239.55   

                     Global_intensity  Sub_metering_1  Sub_metering_2  \
datetime                                                                
2006-12-16 17:24:00              18.4             0.0             1.0   
2006-12-16 17:25:00              23.0             0.0             1.0   
2006-12-16 17:26:00              23.0             0.0             2.0   
2006-12-16 17:27:00              23.0             0.0             1.0   
2006-12-16 17:28:00              15.8             0.0             1.0   
...                               ...             ...             ...   
2010-11-26 20:58:00               4.0             0.0             0.0   
2010-11-26 20:59:00               4.0             0.0             0.0   
2010-11-26 21:00:00               3.8             0.0             0.0   
2010-11-26 21:01:00               3.8             0.0             0.0   
2010-11-26 21:02:00               3.8             0.0             0.0   

                     Sub_metering_3  
datetime                             
2006-12-16 17:24:00            17.0  
2006-12-16 17:25:00            16.0  
2006-12-16 17:26:00            17.0  
2006-12-16 17:27:00            17.0  
2006-12-16 17:28:00            17.0  
...                             ...  
2010-11-26 20:58:00             0.0  
2010-11-26 20:59:00             0.0  
2010-11-26 21:00:00             0.0  
2010-11-26 21:01:00             0.0  
2010-11-26 21:02:00             0.0  

[2049280 rows x 7 columns]
```

## Prompt Used

```bash
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

    Provide me with only python pandas code for below natural language questions, please only provide the pandas code  which can provide the result when executed:
    {natural_language_question}
    """
```
## Fetching response from LLM
```bash
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
```

## Execution of Code generated by LLM

```bash
def execute_query(query,df):
    df_info = {'columns': df.columns.tolist()}
    code = generate_pandas_code(query,df_info)

    # Clean the code by removing markdown formatting so that it can be executed
    if code:
        code_clean = re.sub(r"```(?:python)?\n?|```", "", code).strip()
    try:
        exec(code_clean,{'df':df}) #execute the code in the context of the dataframe
    except Exception as e:
        print(f"Error executing code: {e}")
``` 
