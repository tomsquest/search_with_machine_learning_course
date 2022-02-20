from flask import g
from opensearchpy import OpenSearch

def get_opensearch():
    if 'opensearch' not in g:
        g.opensearch = OpenSearch(
            hosts = [{'host': 'localhost', 'port': 9200}],
            http_compress = True,
            http_auth = ('admin', 'admin'),
            use_ssl = True,
            verify_certs = False,
            ssl_assert_hostname = False,
            ssl_show_warn = False,
        )

    return g.opensearch
