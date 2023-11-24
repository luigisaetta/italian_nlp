import os
import pandas as pd
import streamlit as st
from collections import Counter

from sentiment import SentimentAnalyzer

label_list = ["negativo","neutro","positivo"]

replacement_mapping = {
            'negativo': 0,
            'neutro': 1,
            'positivi': 2
            }

def remove_comma(text):
    text = text.replace(",", " ")

    return text

def remove_newline(text):
    text = text.replace("\n", " ")

    return text

def has_duplicates(list_texts):
    item_count = Counter(list_texts)

    has_duplicates = any(count > 1 for count in item_count.values())

    if has_duplicates:
        print("There are duplicates in the list.")
        for item, count in item_count.items():
            if count > 1:
                    print(f"{item} appears {count} times.")
    else:
        print("There are no duplicates in the list.")

    return has_duplicates

def read_text_files(directory):
    messages = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text = file.read().strip()
                text = remove_comma(text)
                text = remove_newline(text)

                messages.append(text)
    
    if has_duplicates(messages):
        # remove duplicates
        messages = list(set(messages))

    return messages

def display_editable_table(messages, option_autolabel):
    st.header("Labelling dati per Sentiment Analysis")

    if option_autolabel:
        print("Call model to init labels...")
        print()
        sent_analyzer = SentimentAnalyzer()

        comp_label_list = []

        for msg in messages:
            comp_label = sent_analyzer.analyze(msg)
            comp_label_list.append(comp_label)

    # Create an empty DataFrame
    if option_autolabel:
        data = {'text': messages, 'label': comp_label_list}
    else:
        data = {'text': messages, 'label': ['neutro'] * len(messages)}
    
    df = pd.DataFrame(data)

    # Creating text input and select box in each row
    for i in range(len(messages)):
        with st.container():
            cols = st.columns(2)
            df.at[i, 'text'] = cols[0].text_area(f"Testo {i+1}", df.at[i, 'text'], key=f"msg_{i+1}", height=75)
            
            if option_autolabel:
                model_gen_index = replacement_mapping[comp_label_list[i]]
                df.at[i, 'label'] = cols[1].selectbox("Sentimento", label_list, key=f"label_{i+1}", index=model_gen_index)
            else:
                #init to neutral
                df.at[i, 'label'] = cols[1].selectbox("Sentimento", label_list, key=f"label_{i+1}", index=1)

    # Displaying the DataFrame
    st.write(df)

    # Button to save the data (optional)
    if st.button('Salvataggio Dati'):
        # Process or save the data
        st.write("Dati salvati")

        #replace with integer (0,1,2)
        df['label'] = df['label'].replace(replacement_mapping)

        df.to_csv("labelled_text.csv", index=None)

# Sidebar for directory input
if 'loaded_messages' not in st.session_state:
    st.session_state['loaded_messages'] = []

st.sidebar.header("Settings")
directory = st.sidebar.text_input("Enter the directory path:", "path_to_your_text_files")

# On enable definition of initial label using Ai Model trained...
option_autolabel = st.sidebar.radio(
    "Autolabel",
    (True, False)
)

# Button to load messages from the directory
if st.sidebar.button("Carica messaggi"):
    if os.path.isdir(directory):
        st.session_state['loaded_messages'] = read_text_files(directory)
    else:
        st.error("Invalid directory path.")
    

# Check if messages are loaded
if st.session_state['loaded_messages']:
    display_editable_table(st.session_state['loaded_messages'], option_autolabel)
else:
    st.write("Enter a directory path in the sidebar and click 'Load Messages'.")
