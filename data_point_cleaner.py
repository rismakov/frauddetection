import pandas as pd
import numpy as np
import string


def clean_data_point(df ):
    num_exclaimation = []
    num_question = []
    num_words = []
    num_upper_words = []
    count = lambda l1,l2: sum([1 for x in l1 if x in l2])
    for desc in df['description']:
        upper = 0
        num_exclaimation.append(count(desc, string.punctuation[0]))
        num_question.append(count(desc, string.punctuation[20]))
        num_words.append(count(desc, " "))
        for word in desc.split(" "):
            if word.isupper():
                upper +=1
        num_upper_words.append(upper)
    df['num_exclaimation'] = num_exclaimation
    df['num_question'] = num_question
    df['num_words'] = num_words
    df['num_upper_words'] = num_upper_words

    df.user_type[df.user_type > 4] = 4

    for col in df.columns:
        if isinstance(df[col][0],int) or isinstance(df[col][0],float):
            df[col].fillna(df[col].mean())

    df = df[['user_age','user_type','body_length','num_question', 'num_words', 'num_upper_words', 'num_exclaimation']]
    return df
