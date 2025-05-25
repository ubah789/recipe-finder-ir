import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your dataset (make sure the file is in the same folder)
df = pd.read_csv("RAW_recipes.csv")

# Fill nulls for consistency
df['ingredients'] = df['ingredients'].fillna("")
df['description'] = df['description'].fillna("")


# Basic Boolean filtering
def boolean_search(query):
    # Split query on AND, OR, NOT
    query = query.lower()
    filtered_df = df.copy()

    if " and " in query:
        parts = query.split(" and ")
        for part in parts:
            filtered_df = filtered_df[filtered_df['ingredients'].str.contains(part.strip(), case=False)]
    elif " or " in query:
        parts = query.split(" or ")
        cond = False
        for part in parts:
            cond |= df['ingredients'].str.contains(part.strip(), case=False)
        filtered_df = df[cond]
    elif " not " in query:
        parts = query.split(" not ")
        include = parts[0].strip()
        exclude = parts[1].strip()
        filtered_df = df[
            df['ingredients'].str.contains(include, case=False) & ~df['ingredients'].str.contains(exclude, case=False)]
    else:
        filtered_df = df[df['ingredients'].str.contains(query.strip(), case=False)]

    return filtered_df


# TF-IDF based ranking
def rank_recipes(filtered_df, query):
    if filtered_df.empty:
        return filtered_df

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(filtered_df['description'])
    query_vector = tfidf.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    filtered_df = filtered_df.copy()
    filtered_df['score'] = similarity_scores
    ranked_df = filtered_df.sort_values(by='score', ascending=False)
    return ranked_df
