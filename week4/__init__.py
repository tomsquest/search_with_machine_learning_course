import os

import \
    fasttext
from flask import Flask
from flask import render_template

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)

        model_path = os.environ.get("MODEL_LOC", "/home/tom/Dev/search_with_machine_learning_course/queries.min100.bin")
        if model_path and os.path.isfile(model_path):
            print("Loading model at %s" % model_path)
            app.config["query_model"] = fasttext.load_model(model_path)
        else:
            print("Model not found at location %s" % model_path)
            exit(1)

        app.config["index_name"] = os.environ.get("INDEX_NAME", "bbuy_products")
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # A simple landing page
    #@app.route('/')
    #def index():
    #    return render_template('index.jinja2')

    from . import search
    app.register_blueprint(search.bp)
    app.add_url_rule('/', view_func=search.query)

    return app
