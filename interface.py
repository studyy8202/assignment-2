from multiprocessing.sharedctypes import Value
import os
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st



npo_df = pd.read_csv('npo_user.csv')

npo_df['user_id'] = npo_df['user_id'].astype(int)



def remove_duplicates_from_dict(dict):
        seen_values = set()
    # List to store keys of duplicate values
        keys_to_remove = []
        for key, value in dict.items():
            # Check if the value is already encountered
            value = tuple(value) if isinstance(value, list) else value
            if value in seen_values:
                # If duplicate, mark the key for removal
                keys_to_remove.append(key)
            else:
                 # Add the value to the set of seen values
                 seen_values.add(value)
    
             # Remove keys with duplicate values from the dictionary
        for key in keys_to_remove:
            del dict[key]
# Streamlit layout
st.title("RECOMMENDER SYSTEM FOR NPO")

user_id_str = st.text_input("Enter your user ID")

user_id = None

# Convert the user input to an integer if provided
if user_id_str:
    try:
        user_id = int(user_id_str)
    except valueError:
        st.error("Please enter a valid integer for user ID.")

selected_option = st.sidebar.selectbox("Select an option", ["Recently Watched", "For you", "You Might Like", "Watchlist", "Search"])

# Display only 10 movies initially-
num_movies_to_display = min(len(npo_df), 10)

# Set to keep track of unique series names
unique_series = npo_df[npo_df['user_id'] == user_id]['Serie']


#### personas ####
personas = {'Ambitieuze Jongeren': {
  'Gezondheid/opvoeding': 0.01587301587301588,
  'Human Interest': 0.01587301587301588,
  'Nieuws/actualiteiten': 0.01587301587301588,
  'Natuur': 0.01587301587301588,
  'Reizen': 0.01587301587301588,
  'Geschiedenis': 0.01587301587301588,
  'Politiek': 0.00793650793650794,
  'Documentaire': 0.01587301587301588,
  'Drama': 0.04761904761904762,
  'Levenswijze': 0.03174603174603176,
  'Consumenten informatie': 0.00793650793650794,
  'Familie': 0.03174603174603176,
  'Biografie': 0.01587301587301588,
  'Informatief': 0.01587301587301588,
  'Reality TV': 0.05555555555555556,
  'Onderzoeksjournalistiek': 0.00793650793650794,
  'Amusement': 0.04761904761904762,
  'Komisch/Satire': 0.05555555555555556,
  'Science Fiction': 0.04761904761904762,
  'Spanning': 0.04761904761904762,
  'Horror': 0.04761904761904762,
  'Muziek': 0.04761904761904762,
  'Reportage': 0.01587301587301588,
  '(sub)culturen': 0.00793650793650794,
  'Sport': 0.04761904761904762,
  'Liefde': 0.04761904761904762,
  'Spel/quiz': 0.04761904761904762,
  'Maatschappelijk Debat': 0.00793650793650794,
  'Serie': 0.03174603174603176,
  'Kunst/cultuur': 0.01587301587301588,
  'Cabaret': 0.03174603174603176,
  'Sport - informatie': 0.01587301587301588,
  'Wetenschap': 0.03174603174603176,
  'Interviewprogramma/hosted show': 0.04761904761904762,
  'Religieus': 0.01587301587301588,
  'Komisch': 0.00793650793650794,
  'Broadcasters':{
        'AVROTROS': 0.15,
        'PowNed': 0.1,
        'MAX': 0.025,
        'WNL': 0.025,
        'VPRO': 0.15,
        'HUMAN': 0.05,
        'BNNVARA': 0.15,
        'KRO-NCRV': 0.1,
        'EO': 0.05,
        'Omroepvereniging Ongehoord Nederland': 0.05,
        'ZWART': 0.05,
        'NOS': 0.05,
        'NTR': 0.05
    }
  },


 'Avontuurlijke Stadsbewoners': {
  'Gezondheid/opvoeding': 0.022099447513812157,
  'Human Interest': 0.03314917127071824,
  'Nieuws/actualiteiten': 0.03867403314917127,
  'Natuur': 0.03867403314917127,
  'Reizen': 0.03867403314917127,
  'Geschiedenis': 0.03867403314917127,
  'Politiek': 0.03867403314917127,
  'Documentaire': 0.03314917127071824,
  'Drama': 0.022099447513812157,
  'Levenswijze': 0.03314917127071824,
  'Consumenten informatie': 0.011049723756906079,
  'Familie': 0.022099447513812157,
  'Biografie': 0.022099447513812157,
  'Informatief': 0.022099447513812157,
  'Reality TV': 0.022099447513812157,
  'Onderzoeksjournalistiek': 0.03314917127071824,
  'Amusement': 0.022099447513812157,
  'Komisch/Satire': 0.022099447513812157,
  'Science Fiction': 0.022099447513812157,
  'Spanning': 0.022099447513812157,
  'Horror': 0.022099447513812157,
  'Muziek': 0.03314917127071824,
  'Reportage': 0.03314917127071824,
  '(sub)culturen': 0.03314917127071824,
  'Sport': 0.03314917127071824,
  'Liefde': 0.022099447513812157,
  'Spel/quiz': 0.011049723756906079,
  'Maatschappelijk Debat': 0.03314917127071824,
  'Serie': 0.022099447513812157,
  'Kunst/cultuur': 0.03314917127071824,
  'Cabaret': 0.03314917127071824,
  'Sport - informatie': 0.03314917127071824,
  'Wetenschap': 0.03314917127071824,
  'Interviewprogramma/hosted show': 0.022099447513812157,
  'Religieus': 0.03314917127071824,
  'Komisch': 0.011049723756906079,
  'Broadcasters':{
        'AVROTROS': 0.1,
        'PowNed': 0.1,
        'MAX': 0.025,
        'WNL': 0.05,
        'VPRO': 0.15,
        'HUMAN': 0.1,
        'BNNVARA': 0.15,
        'KRO-NCRV': 0.05,
        'EO': 0.05,
        'Omroepvereniging Ongehoord Nederland': 0.025,
        'ZWART': 0.05,
        'NOS': 0.1,
        'NTR': 0.05
    }
  },

 'Zelfbewuste Familiemensen': {'Gezondheid/opvoeding': 0.03763440860215054,
  'Human Interest': 0.03225806451612903,
  'Nieuws/actualiteiten': 0.03763440860215054,
  'Natuur': 0.021505376344086027,
  'Reizen': 0.03225806451612903,
  'Geschiedenis': 0.03763440860215054,
  'Politiek': 0.03763440860215054,
  'Documentaire': 0.03225806451612903,
  'Drama': 0.021505376344086027,
  'Levenswijze': 0.03763440860215054,
  'Consumenten informatie': 0.021505376344086027,
  'Familie': 0.03763440860215054,
  'Biografie': 0.03225806451612903,
  'Informatief': 0.03225806451612903,
  'Reality TV': 0.021505376344086027,
  'Onderzoeksjournalistiek': 0.03225806451612903,
  'Amusement': 0.03225806451612903,
  'Komisch/Satire': 0.021505376344086027,
  'Science Fiction': 0.021505376344086027,
  'Spanning': 0.021505376344086027,
  'Horror': 0.021505376344086027,
  'Muziek': 0.03225806451612903,
  'Reportage': 0.021505376344086027,
  '(sub)culturen': 0.021505376344086027,
  'Sport': 0.03225806451612903,
  'Liefde': 0.021505376344086027,
  'Spel/quiz': 0.021505376344086027,
  'Maatschappelijk Debat': 0.03225806451612903,
  'Serie': 0.021505376344086027,
  'Kunst/cultuur': 0.021505376344086027,
  'Cabaret': 0.03225806451612903,
  'Sport - informatie': 0.03225806451612903,
  'Wetenschap': 0.021505376344086027,
  'Interviewprogramma/hosted show': 0.03225806451612903,
  'Religieus': 0.021505376344086027,
  'Komisch': 0.010752688172043013,
  'Broadcasters' : {
        'AVROTROS': 0.1,
        'PowNed': 0.025,
        'MAX': 0.05,
        'WNL': 0.15,
        'VPRO': 0.15,
        'HUMAN': 0.1,
        'BNNVARA': 0.1,
        'KRO-NCRV': 0.05,
        'EO': 0.1,
        'Omroepvereniging Ongehoord Nederland': 0.05,
        'ZWART': 0.025,
        'NOS': 0.05,
        'NTR': 0.05
    }
  },



 'Technische Doeners': {
  'Gezondheid/opvoeding': 0.014184397163120569,
  'Human Interest': 0.014184397163120569,
  'Nieuws/actualiteiten': 0.028368794326241138,
  'Natuur': 0.014184397163120569,
  'Reizen': 0.028368794326241138,
  'Geschiedenis': 0.028368794326241138,
  'Politiek': 0.028368794326241138,
  'Documentaire': 0.028368794326241138,
  'Drama': 0.028368794326241138,
  'Levenswijze': 0.028368794326241138,
  'Consumenten informatie': 0.028368794326241138,
  'Familie': 0.028368794326241138,
  'Biografie': 0.028368794326241138,
  'Informatief': 0.0425531914893617,
  'Reality TV': 0.028368794326241138,
  'Onderzoeksjournalistiek': 0.028368794326241138,
  'Amusement': 0.028368794326241138,
  'Komisch/Satire': 0.028368794326241138,
  'Science Fiction': 0.028368794326241138,
  'Spanning': 0.028368794326241138,
  'Horror': 0.028368794326241138,
  'Muziek': 0.028368794326241138,
  'Reportage': 0.028368794326241138,
  '(sub)culturen': 0.028368794326241138,
  'Sport': 0.04964539007092198,
  'Liefde': 0.014184397163120569,
  'Spel/quiz': 0.028368794326241138,
  'Maatschappelijk Debat': 0.028368794326241138,
  'Serie': 0.028368794326241138,
  'Kunst/cultuur': 0.007092198581560284,
  'Cabaret': 0.028368794326241138,
  'Sport - informatie': 0.04964539007092198,
  'Wetenschap': 0.04964539007092198,
  'Interviewprogramma/hosted show': 0.028368794326241138,
  'Religieus': 0.028368794326241138,
  'Komisch': 0.007092198581560284,
  'Broadcasters':{
        'AVROTROS': 0.1,
        'PowNed': 0.1,
        'MAX': 0.05,
        'WNL': 0.15,
        'VPRO': 0.1,
        'HUMAN': 0.05,
        'BNNVARA': 0.05,
        'KRO-NCRV': 0.05,
        'EO': 0.1,
        'Omroepvereniging Ongehoord Nederland': 0.1,
        'ZWART': 0.0,
        'NOS': 0.1,
        'NTR': 0.05
    }
  },

 'Zorgzame Duizendpoten': {'Gezondheid/opvoeding': 0.049295774647887314,
  'Human Interest': 0.049295774647887314,
  'Nieuws/actualiteiten': 0.028169014084507043,
  'Natuur': 0.028169014084507043,
  'Reizen': 0.049295774647887314,
  'Geschiedenis': 0.014084507042253521,
  'Politiek': 0.014084507042253521,
  'Documentaire': 0.014084507042253521,
  'Drama': 0.028169014084507043,
  'Levenswijze': 0.014084507042253521,
  'Consumenten informatie': 0.014084507042253521,
  'Familie': 0.049295774647887314,
  'Biografie': 0.014084507042253521,
  'Informatief': 0.007042253521126761,
  'Reality TV': 0.049295774647887314,
  'Onderzoeksjournalistiek': 0.014084507042253521,
  'Amusement': 0.04225352112676056,
  'Komisch/Satire': 0.049295774647887314,
  'Science Fiction': 0.014084507042253521,
  'Spanning': 0.028169014084507043,
  'Horror': 0.028169014084507043,
  'Muziek': 0.028169014084507043,
  'Reportage': 0.014084507042253521,
  '(sub)culturen': 0.014084507042253521,
  'Sport': 0.04225352112676056,
  'Liefde': 0.04225352112676056,
  'Spel/quiz': 0.04225352112676056,
  'Maatschappelijk Debat': 0.014084507042253521,
  'Serie': 0.028169014084507043,
  'Kunst/cultuur': 0.014084507042253521,
  'Cabaret': 0.028169014084507043,
  'Sport - informatie': 0.04225352112676056,
  'Wetenschap': 0.007042253521126761,
  'Interviewprogramma/hosted show': 0.04225352112676056,
  'Religieus': 0.028169014084507043,
  'Komisch': 0.014084507042253521,
  'Broadcasters':{
        'AVROTROS': 0.15,
        'PowNed': 0.05,
        'MAX': 0.1,
        'WNL': 0.1,
        'VPRO': 0.1,
        'HUMAN': 0.05,
        'BNNVARA': 0.1,
        'KRO-NCRV': 0.1,
        'EO': 0.1,
        'Omroepvereniging Ongehoord Nederland': 0.05,
        'ZWART': 0.0,
        'NOS': 0.05,
        'NTR': 0.05
    }
  },

 'Authentieke Gelovigen': {
  'Gezondheid/opvoeding': 0.028571428571428574,
  'Human Interest': 0.028571428571428574,
  'Nieuws/actualiteiten': 0.049999999999999996,
  'Natuur': 0.028571428571428574,
  'Reizen': 0.028571428571428574,
  'Geschiedenis': 0.028571428571428574,
  'Politiek': 0.028571428571428574,
  'Documentaire': 0.028571428571428574,
  'Drama': 0.028571428571428574,
  'Levenswijze': 0.028571428571428574,
  'Consumenten informatie': 0.028571428571428574,
  'Familie': 0.028571428571428574,
  'Biografie': 0.028571428571428574,
  'Informatief': 0.028571428571428574,
  'Reality TV': 0.028571428571428574,
  'Onderzoeksjournalistiek': 0.028571428571428574,
  'Amusement': 0.028571428571428574,
  'Komisch/Satire': 0.028571428571428574,
  'Science Fiction': 0.014285714285714287,
  'Spanning': 0.028571428571428574,
  'Horror': 0.014285714285714287,
  'Muziek': 0.028571428571428574,
  'Reportage': 0.028571428571428574,
  '(sub)culturen': 0.028571428571428574,
  'Sport': 0.028571428571428574,
  'Liefde': 0.028571428571428574,
  'Spel/quiz': 0.028571428571428574,
  'Maatschappelijk Debat': 0.028571428571428574,
  'Serie': 0.028571428571428574,
  'Kunst/cultuur': 0.028571428571428574,
  'Cabaret': 0.0071428571428571435,
  'Sport - informatie': 0.028571428571428574,
  'Wetenschap': 0.0071428571428571435,
  'Interviewprogramma/hosted show': 0.028571428571428574,
  'Religieus': 0.028571428571428574,
  'Komisch': 0.049999999999999996,
  'Broadcasters':{
        'AVROTROS': 0.1,
        'PowNed': 0.025,
        'MAX': 0.1,
        'WNL': 0.15,
        'VPRO': 0.1,
        'HUMAN': 0.025,
        'BNNVARA': 0.025,
        'KRO-NCRV': 0.15,
        'EO': 0.15,
        'Omroepvereniging Ongehoord Nederland': 0.1,
        'ZWART': 0.0,
        'NOS': 0.05,
        'NTR': 0.025
    }
  },


 'De Welgestelde Verdiepingszoekers': {
  'Gezondheid/opvoeding': 0.023255813953488375,
  'Human Interest': 0.03488372093023256,
  'Nieuws/actualiteiten': 0.040697674418604654,
  'Natuur': 0.040697674418604654,
  'Reizen': 0.023255813953488375,
  'Geschiedenis': 0.040697674418604654,
  'Politiek': 0.040697674418604654,
  'Documentaire': 0.03488372093023256,
  'Drama': 0.023255813953488375,
  'Levenswijze': 0.03488372093023256,
  'Consumenten informatie': 0.03488372093023256,
  'Familie': 0.03488372093023256,
  'Biografie': 0.03488372093023256,
  'Informatief': 0.03488372093023256,
  'Reality TV': 0.011627906976744188,
  'Onderzoeksjournalistiek': 0.03488372093023256,
  'Amusement': 0.011627906976744188,
  'Komisch/Satire': 0.011627906976744188,
  'Science Fiction': 0.011627906976744188,
  'Spanning': 0.011627906976744188,
  'Horror': 0.011627906976744188,
  'Muziek': 0.023255813953488375,
  'Reportage': 0.03488372093023256,
  '(sub)culturen': 0.03488372093023256,
  'Sport': 0.023255813953488375,
  'Liefde': 0.011627906976744188,
  'Spel/quiz': 0.011627906976744188,
  'Maatschappelijk Debat': 0.03488372093023256,
  'Serie': 0.023255813953488375,
  'Kunst/cultuur': 0.040697674418604654,
  'Cabaret': 0.03488372093023256,
  'Sport - informatie': 0.023255813953488375,
  'Wetenschap': 0.040697674418604654,
  'Interviewprogramma/hosted show': 0.023255813953488375,
  'Religieus': 0.03488372093023256,
  'Komisch': 0.023255813953488375,
  'Broadcasters': {
        'AVROTROS': 0.05,
        'PowNed': 0.025,
        'MAX': 0.05,
        'WNL': 0.05,
        'VPRO': 0.15,
        'HUMAN': 0.15,
        'BNNVARA': 0.15,
        'KRO-NCRV': 0.1,
        'EO': 0.05,
        'Omroepvereniging Ongehoord Nederland': 0.0,
        'ZWART': 0.025,
        'NOS': 0.1,
        'NTR': 0.1
    }
  },

 'Behoedzame Senioren': {'Gezondheid/opvoeding': 0.03125,
  'Human Interest': 0.03125,
  'Nieuws/actualiteiten': 0.05468749999999999,
  'Natuur': 0.03125,
  'Reizen': 0.03125,
  'Geschiedenis': 0.03125,
  'Politiek': 0.015625,
  'Documentaire': 0.03125,
  'Drama': 0.015625,
  'Levenswijze': 0.03125,
  'Consumenten informatie': 0.03125,
  'Familie': 0.05468749999999999,
  'Biografie': 0.03125,
  'Informatief': 0.03125,
  'Reality TV': 0.0078125,
  'Onderzoeksjournalistiek': 0.03125,
  'Amusement': 0.03125,
  'Komisch/Satire': 0.015625,
  'Science Fiction': 0.0078125,
  'Spanning': 0.0078125,
  'Horror': 0.0078125,
  'Muziek': 0.015625,
  'Reportage': 0.03125,
  '(sub)culturen': 0.03125,
  'Sport': 0.03125,
  'Liefde': 0.015625,
  'Spel/quiz': 0.015625,
  'Maatschappelijk Debat': 0.03125,
  'Serie': 0.03125,
  'Kunst/cultuur': 0.03125,
  'Cabaret': 0.03125,
  'Sport - informatie': 0.03125,
  'Wetenschap': 0.03125,
  'Interviewprogramma/hosted show': 0.03125,
  'Religieus': 0.03125,
  'Komisch': 0.04687499999999999,
  'Broadcasters': {
        'AVROTROS': 0.1,
        'PowNed': 0.025,
        'MAX': 0.15,
        'WNL': 0.15,
        'VPRO': 0.05,
        'HUMAN': 0.025,
        'BNNVARA': 0.05,
        'KRO-NCRV': 0.1,
        'EO': 0.1,
        'Omroepvereniging Ongehoord Nederland': 0.1,
        'ZWART': 0.0,
        'NOS': 0.1,
        'NTR': 0.05
    }

  }}

#### fairness-accountability recommendations ####

nltk.download('stopwords')
dutch_stopwords = stopwords.words('dutch')

# vectorize articles based on their title
vectorizer = TfidfVectorizer(strip_accents='unicode', lowercase=True, stop_words=dutch_stopwords)
articles_vectors = vectorizer.fit_transform(npo_df['Content'])

# calculate cosine similarity between article vectors and create similarity matrix
sim_matrix = cosine_similarity(articles_vectors, articles_vectors)

sim_df = pd.DataFrame(sim_matrix, index=npo_df.index, columns=npo_df.index)


# create function that gets similarity scores for a specific article
def get_similarities(serie_id):
    similarities = sim_df.loc[serie_id]
    # drop similarity with the same article
    #similarities = similarities.drop(video_id, inplace=True)
    return similarities.rename('similarity').to_frame()

# create function that adds similarity scores to complete dataframe
def recommend_videos(serie_ids):
    recommendations = pd.DataFrame()
    for serie_id in serie_ids:
        similarities = get_similarities(serie_id)
        df_npo = npo_df.drop(serie_id, axis='rows').join(similarities)
        recommendations = pd.concat([recommendations, df_npo])
    return recommendations.sort_values('similarity', ascending=False)

# create filter function that removes recommendations that are politically polarizing to user
def filter_higher(user_rating, recommendations):
    # if user has a right political score
    if user_rating > 1:
        filtered_df = recommendations[recommendations['polarization_score'] <= user_rating + 1]
    # if user has a left political score
    elif user_rating < -1:
        filtered_df = recommendations[recommendations['polarization_score'] >= user_rating - 1]
    # if user has centre political score
    else:
        filtered_df = recommendations
    return filtered_df

# function that removes politically distant recommendations to user
def filter_buffer(user_rating, recommendations):
    filtered_df = recommendations[abs(recommendations['polarization_score'] - user_rating) < 5]
    return filtered_df

# function that adds polarization score and accountability score to dataframe
def accountability_score(df_recommendations):
    # accountability score of a recommendation is weighted score of similarity and polarization
    df_recommendations['accountability_score'] = df_recommendations['similarity'] * df_recommendations['inverted_polarization_score']
    return df_recommendations

def get_accountable_recommendations(npo_df, user_id, df_recommendations):
    # get user information of specific user
    user_info = npo_df[npo_df['user_id'] == user_id]

    # get average political score of user
    user_rating = user_info.iloc[0]['political_rating']

    # filter recommendations so they are not politically polarizing for the user
    filtered_recs = filter_higher(user_rating, df_recommendations)

    # filter recommendatiosn so they are in range of political spectrum of user
    filtered_recs = filter_buffer(user_rating, filtered_recs)

    # create accountability score for all remaining recommendations
    accountable_recommendations = accountability_score(filtered_recs)

    # sort articles
    accountable_recommendations = accountable_recommendations.sort_values('accountability_score', ascending=False)

    return accountable_recommendations

# operationalization of fairness
def get_recommendations_fairness(df_recommendations, npo_df, user_id, personas):

    # Get relevant broadcasters and genres
    persona = npo_df.loc[npo_df['user_id'] == user_id, 'persona'].values[0]
    persona = personas[persona]
    genres = set([value for key, value in persona.items() if key != 'Broadcasters'])
    sorted_values = sorted(genres)
    list_genres = [key for key, value in persona.items() if value in sorted_values[-2:]]

    broadcasters = persona['Broadcasters']
    list_broadcasters = [key for key, value in broadcasters.items() if value >= 0.1]

    # keep recommendations that are fair in regards to their original persona
    df_fair_recommendations = df_recommendations[df_recommendations['Broadcaster'].isin(list_broadcasters)
                                                 & (df_recommendations['Genre_1'].isin(list_genres)
                                                     | df_recommendations['Genre_2'].isin(list_genres))]

    # Check if there are any movies in the relevant genres
    if not df_fair_recommendations.empty:
        # Select top-ranked movies within relevant genres
        fair_recommendations = df_fair_recommendations
    else:
      fair_recommendations = df_recommendations #regular recommendation if no relevant genres are matched
    return fair_recommendations

#### fairness-serendipity recommendation ####
# operationalization of serendipity
# serendipity = relevance * unexpectedness
# relevance: all unseen shows
# unexpectedness: (1 - average similarity score for each movie to the users history) * (0.5 if broadcaster is not in users preference + 0.5 if genre is not in users preference)

def get_recommendations_serendipity(df_recommendations, npo_df, user_id, sim_df, personas):

    # get the user's watched videos
    watched_videos = npo_df[npo_df['user_id'] == user_id]['series_id'].tolist()

    # measure average similarity score for unseen shows
    average_sim_df = pd.DataFrame()

    for i, watched_video in enumerate(watched_videos):
        similarities = sim_df.loc[watched_video]
        col = f'col{i}'
        average_sim_df[col] = similarities
    average_sim_df['average'] = average_sim_df.mean(axis=1)

# Calculate the average similarity score
    average_sim_df['average'] = average_sim_df.mean(axis=1)

    # calculate unexpectedness
    df_recommendations['unexpectedness'] = 1 - average_sim_df['average']

    # get list of expected genres
    persona = npo_df.loc[npo_df['user_id'] == user_id, 'persona'].values[0]
    persona = personas[persona]
    genres = set([value for key, value in persona.items() if key != 'Broadcasters'])
    sorted_values = sorted(genres)
    list_genres = [key for key, value in persona.items() if value in sorted_values[-2:]]

    # get list of expected broadcasters
    broadcasters = persona['Broadcasters']
    list_broadcasters = [key for key, value in broadcasters.items() if value >= 0.1]

    df_recommendations['preferences'] = 0
    df_recommendations.loc[~df_recommendations['Broadcaster'].isin(list_broadcasters), 'preferences'] += 0.5
    df_recommendations.loc[~df_recommendations['Genre_1'].isin(list_genres) | ~df_recommendations['Genre_2'].isin(list_genres), 'preferences'] += 0.5

    df_recommendations['unexpectedness'] = df_recommendations['unexpectedness'] * df_recommendations['preferences']

    # include relevance into dataframe
    df_recommendations['relevance'] = 1 * df_recommendations['similarity']
    df_recommendations.loc[df_recommendations['series_id'].isin(watched_videos), 'relevance'] = 0

    # # apply weights to the relevance and unexpectedness scores
    # weighted_relevance = df_npo['relevance'] * relevance_weight
    # weighted_unexpectedness = df_npo['unexpectedness'] * unexpectedness_weight

    # add serendipity to dataframe and remove irrelevant videos
    df_recommendations['serendipity'] = df_recommendations['unexpectedness'] * df_recommendations['relevance']
    df_recommendations = df_recommendations[df_recommendations['serendipity'] != 0]

    df_serendipity = df_recommendations.sort_values(by='serendipity', ascending=False)

    return df_serendipity

#### interface ####
# Check if user ID is provided
if user_id is not None:
  if selected_option == "Recently Watched":  
    st.subheader("Your recently watched shows")

    for series in unique_series:
        series_data = npo_df[npo_df["Serie"] == series].head(1)
        cols = st.columns(2)
        for index, row in series_data.iterrows():
            with cols[0]:
                st.header(row["Serie"])
                st.image(row["Image_serie"], width=300)
            with cols[1] :
                st.header(row["Broadcaster"])
                st.write(f"Genre: {row['Genre_1']} | {row['Genre_2']}")            
            with st.expander("Episodes"):
                unique_episodes = sorted(npo_df[npo_df["Serie"] == series]["Episode"].unique(), reverse=True)
                # Iterate over unique episodes in descending order
                for episode_number in unique_episodes:
                     episode_row = npo_df[(npo_df["Serie"] == series) & (npo_df["Episode"] == episode_number)].iloc[0]
                     st.subheader(f"Episode {episode_row['Episode']}")
                     st.image(episode_row["Image_ep"], width=300)
                     st.text(f"Title: {episode_row['Title']}")
                     st.write(f"Description: {episode_row['Description']}", height=100) 
  elif selected_option == "For you": 
    st.subheader("Shows that are tailored for you")  
    # get accountable recommendations
    df_recommendations = recommend_videos(npo_df[npo_df['user_id'] == user_id]['series_id'].tolist())
    df_recommendations = get_accountable_recommendations(npo_df, user_id, df_recommendations)
    df_recommendations_fairness = get_recommendations_fairness(df_recommendations, npo_df, user_id, personas) 
    fairness_unique = df_recommendations_fairness["Serie"].unique()
    for series in fairness_unique[:20]:
        series_data = npo_df[npo_df["Serie"] == series].head(1)
        cols = st.columns(2)
        for index, row in series_data.iterrows():
            with cols[0]:
                st.header(row["Serie"])
                st.image(row["Image_serie"], width=300)
            with cols[1] :
                st.header(row["Broadcaster"])
                st.write(f"Genre: {row['Genre_1']} | {row['Genre_2']}") 
                button =st.button(':heavy_plus_sign:', key = f"{index}")
                if button:
                    for index, row in series_data.iterrows():
                        key = f'key_{index}{user_id}'  # Generating keys dynamically
                        st.session_state[key] = row["Serie"]
            with st.expander("Episodes"):
                unique_episodes = sorted(npo_df[npo_df["Serie"] == series]["Episode"].unique(), reverse=True)
                # Iterate over unique episodes in descending order
                for episode_number in unique_episodes:
                     episode_row = npo_df[(npo_df["Serie"] == series) & (npo_df["Episode"] == episode_number)].iloc[0]
                     st.subheader(f"Episode {episode_row['Episode']}")
                     st.image(episode_row["Image_ep"], width=300)
                     st.text(f"Title: {episode_row['Title']}")
                     st.write(f"Description: {episode_row['Description']}", height=100)
    feedback = st.radio('Feedback',
    [':+1:', ':-1:'], index = None, key ='feedback') 
    st.session_state.feedback 
  
  elif selected_option == "You Might Like":
    st.subheader("Shows that you might find interesting")  
    df_recommendations = recommend_videos(npo_df[npo_df['user_id'] == user_id]['series_id'].tolist())
    df_recommendations = get_accountable_recommendations(npo_df, user_id, df_recommendations)
    df_recommendations_serendipity = get_recommendations_serendipity(df_recommendations, npo_df, user_id, sim_df, personas)  
    serendipity_unique = df_recommendations_serendipity["Serie"].unique()
    for series in serendipity_unique[:20]:
        series_data = npo_df[npo_df["Serie"] == series].head(1)
        cols = st.columns(2)
        for index, row in series_data.iterrows():
            with cols[0]:
                st.header(row["Serie"])
                st.image(row["Image_serie"], width=300)
            with cols[1] :
                st.header(row["Broadcaster"])
                st.write(f"Genre: {row['Genre_1']} | {row['Genre_2']}")
                button =st.button(':heavy_plus_sign:', key = f"{index}")
                if button:
                    for index, row in series_data.iterrows():
                        key = f'key_{index}{user_id}'  # Generating keys dynamically
                        st.session_state[key] = row["Serie"]            
            with st.expander("Episodes"):
                unique_episodes = sorted(npo_df[npo_df["Serie"] == series]["Episode"].unique(), reverse=True)
                # Iterate over unique episodes in descending order
                for episode_number in unique_episodes:
                     episode_row = npo_df[(npo_df["Serie"] == series) & (npo_df["Episode"] == episode_number)].iloc[0]
                     st.subheader(f"Episode {episode_row['Episode']}")
                     st.image(episode_row["Image_ep"], width=300)
                     st.text(f"Title: {episode_row['Title']}")
                     st.write(f"Description: {episode_row['Description']}", height=100)
    feedback = st.radio('Feedback',
    [':+1:', ':-1:'], index = None, key ='feedback') 
    st.session_state.feedback 

  elif selected_option == "Search":
    show_input = st.text_input("Enter name of the show", key= "show_input")
    show_name = st.session_state['show_input']   
    if show_input:
                show_name = st.session_state['show_input']
                series_data = npo_df[npo_df["Serie"] == show_name].head(1)
                cols = st.columns(2)
                for index, row in series_data.iterrows():
                    with cols[0]:
                        st.header(row["Serie"])
                        st.image(row["Image_serie"], width=300)
                    with cols[1]:
                        st.header(row["Broadcaster"])
                        st.write(f"Genre: {row['Genre_1']} | {row['Genre_2']}")
                        button =st.button(':heavy_plus_sign:', key = f"add_button_{index}")
                        if button:
                            for index, row in series_data.iterrows():
                                key = f'watchlist_{index}{user_id}'  # Generating keys dynamically
                                st.session_state[key] = row["Serie"]
                    with st.expander("Episodes"):
                        unique_episodes = sorted(npo_df[npo_df["Serie"] == show_name]["Episode"].unique(), reverse=True)
                        # Iterate over unique episodes in descending order
                        for episode_number in unique_episodes:
                            episode_row = npo_df[(npo_df["Serie"] == show_name) & (npo_df["Episode"] == episode_number)].iloc[0]
                            st.subheader(f"Episode {episode_row['Episode']}")
                            st.image(episode_row["Image_ep"], width=300)
                            st.text(f"Title: {episode_row['Title']}")
                            st.write(f"Description: {episode_row['Description']}", height=100)
  else:
    st.subheader('Your watchlist')
    user_string = str(user_id)
    # Retrieve or create the user-specific watchlist based on user_id
    for key, value in st.session_state.items():
       if key.startswith('watchlist_'):# Check for keys starting with 'watchlist_'
         if user_string in key:   
          if st.session_state[key]:   
            show_name = st.session_state[key]  # Retrieve the movie name from the session state using the button key
            series_data = npo_df[npo_df["Serie"] == show_name].head(1)
            cols = st.columns(2)
            for index, row in series_data.iterrows():
             with cols[0]:
                st.header(row["Serie"])
                st.image(row["Image_serie"], width=300)
             with cols[1]:
                st.header(row["Broadcaster"])
                st.write(f"Genre: {row['Genre_1']} | {row['Genre_2']}")
             with st.expander("Episodes"):
                unique_episodes = sorted(npo_df[npo_df["Serie"] == show_name]["Episode"].unique(), reverse=True)
                for episode_number in unique_episodes:
                    episode_row = npo_df[(npo_df["Serie"] == show_name) & (npo_df["Episode"] == episode_number)].iloc[0]
                    st.subheader(f"Episode {episode_row['Episode']}")
                    st.image(episode_row["Image_ep"], width=300)
                    st.text(f"Title: {episode_row['Title']}")
                    st.write(f"Description: {episode_row['Description']}", height=100)

       elif key.startswith('key_'):# Check for keys starting with 'add_button_'
         if user_string in key:
          if st.session_state[key]:    
            movie_name = st.session_state[key]  # Retrieve the movie name from the session state using the button key
            series_data = npo_df[npo_df["Serie"] == movie_name].head(1)
            cols = st.columns(2)
            for _, row in series_data.iterrows():
                with cols[0]:
                    st.header(row["Serie"])
                    st.image(row["Image_serie"], width=300)
                with cols[1]:
                    st.header(row["Broadcaster"])
                    st.write(f"Genre: {row['Genre_1']} | {row['Genre_2']}")
                with st.expander("Episodes"):
                    unique_episodes = sorted(npo_df[npo_df["Serie"] == movie_name]["Episode"].unique(), reverse=True)
                    for episode_number in unique_episodes:
                        episode_row = npo_df[(npo_df["Serie"] == movie_name) & (npo_df["Episode"] == episode_number)].iloc[0]
                        st.subheader(f"Episode {episode_row['Episode']}")
                        st.image(episode_row["Image_ep"], width=300)
                        st.text(f"Title: {episode_row['Title']}")
                        st.write(f"Description: {episode_row['Description']}", height=100)
    
    

               
    




    

elif user_id is not None:
    st.write("No recently watched shows found for the provided user ID.")


