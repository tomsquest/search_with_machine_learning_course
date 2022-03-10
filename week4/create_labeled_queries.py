import argparse
import xml.etree.ElementTree as ET
import re
import pandas as pd
import csv

# Useful if you want to perform stemming.
import nltk
from nltk import \
    word_tokenize
from nltk.corpus import \
    stopwords

stemmer = nltk.stem.PorterStemmer()
stop_words = set(stopwords.words('english'))

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/labeled_query_data.min1000.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,
                     help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name,
                     help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
print("reading categories")
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)),
                          columns=['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
print("reading queries")
df = pd.read_csv(queries_file_name)[['category', 'query']]
df = df[df['category'].isin(categories)]


# Convert queries to lowercase, and optionally implement other normalization, like stemming.
def transform_query(s: str):
    s = s.lower()
    s = re.sub('[\'"]', '', s)
    tokens = word_tokenize(s)
    tokens = [token for token in tokens if not token in stop_words]
    tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(tokens)


# Transform queries in dataframe
print("transforming queries")
df["query"] = df['query'].apply(transform_query)


# Return the category_code of the parent of category_code, or the root
# category if not parent
def get_parent(category_code: str):
    if category_code == "cat00000":
        return category_code
    # print("get_parent of", category_code)
    return \
        parents_df.loc[parents_df['category'] == category_code][
            "parent"].values[0]


# Given some narrow categories, take parent
# Do that a maximum of N times
# Compute how many categories are too narrow
min_queries = 1000
max_iteration = 50
cur_iteration = max_iteration


# Apply function to replace categories by their parent given the category is
# too narrow
def replace_by_parent(series):
    # print("fn", series["category"], series["nb_queries"], min_queries)
    if series["nb_queries"] < min_queries:
        parent = get_parent(series["category"])
        # print("get_parent for", series["category"], " => parent is ", parent)
        return parent
    else:
        # print("> min_queries: ", series["category"])
        return series["category"]


print(f"Starting replacing narrow categories with their parent. Iteration "
      f"{cur_iteration}/{max_iteration}")
while cur_iteration > 0:
    # Add column with number of queries per category
    df["nb_queries"] = df.groupby('category')['query'].transform(len)

    # Find how many categories are too narraw (lesser than min_queries)
    nb_queires_with_narrow_categories = df[df["nb_queries"] < min_queries][
        "category"].count()
    print(f"Found {nb_queires_with_narrow_categories} narrow categories")
    if nb_queires_with_narrow_categories > 0:
        print("Replacing by parents...")
        df["category"] = df.apply(replace_by_parent, axis=1)
        print(f"Iteration {cur_iteration}/{max_iteration} done.")
        cur_iteration -= 1
    else:
        break
print(f"Finished replacing narrow categories. Iteration {cur_iteration}/{max_iteration}")

# # Create labels in fastText format.
df['label'] = '__label__' + df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
# df = df[df['category'].isin(categories)]
df['output'] = df['label'] + ' ' + df['query']
df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\',
                      quoting=csv.QUOTE_NONE, index=False)
print(f"Wrote results to {output_file_name}")
