import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('tweets_preprocessed.csv')


for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna('')
    else:
        df[col] = df[col].fillna(0)

df['text_length'] = df['text'].str.len()

keyword_dummies = pd.get_dummies(df['keyword'], prefix='keyword')
df = pd.concat([df, keyword_dummies], axis=1)


vectorizer = TfidfVectorizer(max_features=500)
text_tfidf = vectorizer.fit_transform(df['text'])
text_tfidf_df = pd.DataFrame(text_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
df = pd.concat([df.reset_index(drop=True), text_tfidf_df.reset_index(drop=True)], axis=1)


csv_path = 'tweets_features.csv'
df.to_csv(csv_path, index=False)