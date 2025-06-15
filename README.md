# Natural Language to Pandas Query Assistant using GROQ LLM

This project evaluates the performance of five open-source Large Language Models (LLMs) in generating Python pandas code from natural language queries. The models are tested on energy data use cases such as computing daily trends, correlations, and plotting patterns for energy disaggregation.

---

## Author

**Your Name:** *Jignesh Rana*  

---

## LLMs used

- **Provider:** [Groq](https://console.groq.com)
- **Model:** `llama-3.3-70b-Versatile, gemma2-9b-it, qwen/qwen3-32b, deepseek-r1-distill-llama-70b, mistral-saba-24b`
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

## Files Included

| File                                  | Description                                              |
| ------------------------------------- | -------------------------------------------------------- |
| `llama_3_3_70b_Versatile.ipynb`       | Responses and analysis using LLaMA                       |
| `gemma2_9b_it.ipynb`                  | Code outputs using Gemma                                 |
| `qwen_qwen3_32b.ipynb`                | Responses from Qwen                                      |
| `deepseek_r1_distill_llama_70b.ipynb` | Code and results from DeepSeek                           |
| `mistral_saba_24b.ipynb`              | Output logs from Mistral                                 |
| `TestingQueries.ipynb`                | Common testing queries for all models                    |
| `llm_results.ipynb`                   | Combined analysis, accuracy comparison, and visual plots |

---
## Querie Evaluated

Each model was prompted to answer:
1. What was the average active power consumption in March 2007?
2. What hour of the day had the highest power usage on Christmas 2006?
3. Compare energy usage (Global_active_power) on weekdays vs weekends.
4. Find days where total energy consumption exceeded 5 kWh. 
5. Plot the energy usage trend for the first week of January 2007. Can you aggregrate data by month.
6. Find the average voltage for each day of the first week of February 2007.
7. What is the correlation between global active power and sub-metering values?
8. What is the correlation between global active power and sub-metering values?. Can you show it using visualization.
9. Plot the energy usage trend for the first week of January 2007. Can you aggregate data by day.
---
## Evaluation Criteria

Each model was scored on:
  - Code generation (was valid Python returned?)
  - Execution (did the code run without error?)
---
## Human Evaluation of LLM Output Quality

- Here I have compared 5 llm models and had presented the results that which model is giving best result for the same set of question and same prompt.
- Here the automatic evaluation was done on only single parameter i.e whether the code is succesfully executing the llm generated code or not.
- But for visual representation, Human evaluation is done where i have personally checked that which llm is generating what kind of plots/results for same set of queries
- My obervation:
  - llama-3.3-70b-Versatile
    -  Strengths: This model generally produced high-quality visual outputs and often provided multiple forms of visualization for enhanced user understanding. It correctly included necessary package imports (matplotlib, seaborn, etc.), which improved execution success.
    -  Weaknesses: In some cases, it generated unnecessary extra plots, adding complexity to the output. Occasionally, while the code was syntactically correct, it failed to print results or call display functions like plt.show(), leading to blank outputs.
    -  Conclusion: Decent for visual queries, but may require light manual editing for non-visual tasks.  
  - deepseek_r1_distill_llama_70b
    - Strengths: This model generally produced high-quality visual outputs and often provided multiple forms of visualization for enhanced user understanding. It correctly included necessary package imports (matplotlib, seaborn, etc.), which improved execution success.
    - Weaknesses: In some cases, it generated unnecessary extra plots, adding complexity to the output. Occasionally, while the code was syntactically correct, it failed to print results or call display functions like plt.show(), leading to blank outputs.
    - Conclusion: Requires significant manual correction; poor reliability for both code and visualization tasks.
  - gemma2-9b-it
    - Strengths: Occasionally produced structurally correct code.
    - Weaknesses: Frequently failed to print the final output or display plots. Even when the logic and syntax were correct, forgetting to include calls like plt.plot() or plt.show() resulted in blank visual outputs, which could confuse users.
    - Conclusion: Lowest performance among the five models. Suitable only with manual correction and debugging.
  - qwen/qwen3-32b
    - Strengths: Generally produced well-structured code with correct imports and output formatting. It handled most visualization tasks adequately.
    - Weaknesses: Minor inconsistencies were noticed in plot results—sometimes the visualizations differed slightly from expectations or used less informative formats.
    - Conclusion: A strong performer, especially for structured data queries.
  - mistral-saba-24b
    - Strengths: This model showed the most precise and accurate code generation. It consistently produced the correct output for both data transformation and visualization tasks, without unnecessary additions. All plots were properly displayed, and outputs were well-formatted.
    - Weaknesses: Very few observed; occasionally minimalistic, but functionally sound.
    - Conclusion: Best overall performer. Highly reliable and clean in response formatting.
---
## Results Summary
![image](https://github.com/user-attachments/assets/d132179d-8e92-4bf7-beb9-b201e06c3681)

![image](https://github.com/user-attachments/assets/fb8ce283-bd2d-45ef-b9c1-5fd6366fd896)

![image](https://github.com/user-attachments/assets/6541950f-891d-4c4b-9a18-70dc81e81e31)
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
prompt = f"""
        You are a Python pandas and matplotlib expert. Given this preprocessed dataframe (index=datetime):

        PREPROCESSING CODE:
    
         df = pd.read_csv('dataset/household_power_consumption.txt',
             	sep=';',
             	parse_dates={{'datetime': {{'Date', 'Time'}}}},
             	infer_datetime_format=True,
             	na_values=['?'],
             	low_memory=False)

        df = df.dropna()
        df['Global_active_power'] = df['Global_active_power'].astype(float)
        df = df.set_index('datetime')
        
        Columns: {df_info_str}
        
        Generate ONLY executable Python/pandas code that:
        1. Answers the question: "{natural_language_question}"
        2. Stores final results in these standard variables when applicable:
           - avg_power_march, max_hour, weekday_avg, weekend_avg 
           - days_exceeded_5_kwh, daily_voltage, correlation
           - plot_data (for visualization outputs only)
        
        ONLY FOR VISUALIZATION QUERIES WHEN QUESTION CONTAINS KEY-WORDS like "plot", "visualize", "graph", "visualization":
            1. MUST include these elements in order:
               a) Data filtering/processing
               b) plt.figure() with size (10,6)
               c) ACTUAL PLOTTING COMMAND (plt.plot, plt.bar, etc.)
               d) Axis labels and title (plt.xlabel, plt.ylabel, plt.title)
               e) Save to "images/{model_name}/[sanitized_title].png"
               f) Buffer handling for plot_data
               g) plt.show()
               h) plt.close()
        
        EXAMPLE FOR QUESTION CONTAIN VISUALIZATION i.e "Plot the energy usage trend for the first week of January 2007."
        ```python
        # Filter data
        filtered = df[(df.index.year==2007) & (df.index.month==1)]
        daily = filtered['Global_active_power'].resample('D').mean()
        
        # Create plot
        plt.figure(figsize=(10,6))
        plt.plot(daily.index, daily, color='blue')  # <-- ACTUAL PLOT COMMAND
        plt.xlabel('Date')
        plt.ylabel('Power (kW)')
        title = 'January 2007 Daily Power Consumption'
        plt.title(title)
        
        # Save and store
        os.makedirs(f"images/{model_name}", exist_ok=True)
        filename = title.lower().replace(' ','_') + '.png'
        plt.savefig(f"images/{model_name}/filename.png")
        buf = io.BytesIO()
        buf.seek(0)
        plot_data = buf.getvalue()
        plt.show()
        plt.close()

        IMPORTANT RULES:
        (.)NEVER skip the actual plotting command (plt.plot, plt.bar, etc.) whereever needed if QUESTION CONTAIN KEY-WORDS.
        (.)show the plot (plt.show()) only for visual questions
        (.)Generate ONLY code - no explanations or markdown formatting
        (.)For non-visual questions, NEVER include plotting code AND ALWAYS PRINT THE FINAL RESULT.
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
# Function to execute the generated pandas code and display the output

def execute_query(query,model_name,df):
    df_info = {'columns': df.columns.tolist()}
    local_vars={}

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
            if line.strip().startswith(("print", "df", "#", "import", "average", "result")) or "=" in line:
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
        exec_vars={'df':df,'plt':plt,'os':os}
        exec(fixed_code, exec_vars,local_vars) #execute the code in the context of the dataframe
         # Check if a plot was attempted but has 0 axes
        if 'plt' in exec_vars:
            fig = plt.gcf()
            if len(fig.axes) == 0 and any(line.strip().startswith(('plt.plot', 'plt.bar', 'plt.figure')) 
                                       for line in fixed_code.split('\n')):
                plt.close()
                print("Plot generation failed - empty figure detected")
                return False
        return True
    except Exception as e:
        print(f"Error executing code: {e}")
        return False
``` 
