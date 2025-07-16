# Convert date column to dotetime
# Filter for 2003 - 2005 data
# Handle duplicates in reviews
# Remove reviews that appear more than 3 times (adjust threshold as needed)
# Newer ratings seem to be more useful than older ones, have higher helpfulness ratio
high_freq_dupes = df['review_body'].value_counts()[df['review_body'].value_counts() > 3].index
df_clean = df[~df['review_body'].isin(high_freq_dupes)]
# Keep only the first review when same user posts identical text
df_clean = df_clean.drop_duplicates(subset=['customer_id', 'review_body'], keep='first')