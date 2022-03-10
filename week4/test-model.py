from xml.etree import \
    ElementTree
import fasttext

query = "sony camera"

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'
# categories_file_name = r'/workspace/datasets/product_data/categories/categories.small.xml'
model_path = "/home/tom/Dev/search_with_machine_learning_course/queries" \
             ".min1000.bin"
model = fasttext.load_model(model_path)


def read_categories():
    tree = ElementTree.parse(categories_file_name)
    root = tree.getroot()
    categories = {}
    for child in root:
        id = child.find('id').text
        name = child.find('name').text
        categories[id] = name
    # print(categories)
    return categories


def predict(query: str, num_predictions=5):
    predictions = model.predict(query, num_predictions)
    # print(predictions)

    category_codes, scores = predictions
    category_codes = [s.replace("__label__", "") for s in category_codes]
    zipped = zip(category_codes, scores)
    return zipped


categories = read_categories()

category_scores = predict(query, 10)

for category_code, score in category_scores:
    category_name = categories.get(category_code, None)
    print(f"{category_code}\t\t{category_name}\t\t\t{score}")
