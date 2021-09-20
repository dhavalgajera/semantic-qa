from elasticsearch import Elasticsearch
import config
es_conn = None
index_name = config.INDEX_NAME


def connect_elastic(ip, port):
    # Connect to an elasticsearch node with the given ip and port
    global es_conn

    es_conn = Elasticsearch([{"host": ip, "port": port}])
    if es_conn.ping():
        print("Connected to elasticsearch...")
    else:
        print("Elasticsearch connection error...")
    return es_conn


def create_qa_index():
    # Define the index mapping
    index_body = {
        "mappings": {
            "properties": {
                "question": {
                    "type": "text"
                },
                "answer": {
                    "type": "text"
                },
                "question_vec": {
                    "type": "dense_vector",
                    "dims": 512
                },
                "answer_vec": {
                    "type": "dense_vector",
                    "dims": 512
                }
            }
        }
    }
    try:
        # Create the index if not exists
        if not es_conn.indices.exists(index_name):
            # Ignore 400 means to ignore "Index Already Exist" error.
            es_conn.indices.create(
                index=index_name, body=index_body  # ignore=[400, 404]
            )
            print("Created Index -> "+index_name)
        else:
            print("Index exists...")
    except Exception as ex:
        print(str(ex))


def insert_qa(body):
    if not es_conn.indices.exists(index_name):
        create_qa_index()
    # Insert a record into the es index
    es_conn.index(index=index_name, body=body)
    # print("QA successfully inserted...")


def semantic_search(query_vec, thresh=1.2, top_n=10):
    # Retrieve top_n semantically similar records for the given query vector
    if not es_conn.indices.exists(index_name):
        return "No records found"
    s_body = {
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "Math.max(cosineSimilarity(params.query_vector, 'question_vec'),cosineSimilarity(params.query_vector, 'answer_vec') )  + 1 ",
                    "params": {"query_vector": query_vec}
                }
            }
        }
    }
    # Semantic vector search with cosine similarity
    result = es_conn.search(index=index_name, body=s_body)
    total_match = len(result["hits"]["hits"])
    print("Total Matches: ", str(total_match))
    # print(result)
    data = []
    if total_match > 0:
        q_ids = []
        for hit in result["hits"]["hits"]:
            if hit['_score'] > thresh and len(data) <= top_n:
                print("--\nscore: {} \n question: {} \n answer: {}\n--".format(
                    hit["_score"], hit["_source"]['question'], hit["_source"]['answer']))

                data.append(
                    {'question': hit["_source"]['question'], 'answer': hit["_source"]['answer'], 'score': hit["_score"]})
    return data


def keyword_search(query, thresh=1.2, top_n=10):
    # Retrieve top_n records using TF-IDF scoring for the given query vector
    if not es_conn.indices.exists(index_name):
        return "No records found"
    k_body = {
        "query": {
            "match": {
                "question": query
            }
        }
    }

    # Keyword search
    result = es_conn.search(index=index_name, body=k_body)
    total_match = len(result["hits"]["hits"])
    print("Total Matches: ", str(total_match))
    # print(result)
    data = []
    if total_match > 0:
        q_ids = []
        for hit in result["hits"]["hits"]:
            # if hit['_score'] > thresh and hit['_source']['q_id'] not in q_ids and len(data) <= top_n:
            if hit['_score'] > thresh and len(data) <= top_n:
                print("--\nscore: {} \n question: {} \n answer: {}\n--".format(
                    hit["_score"], hit["_source"]['question'], hit["_source"]['answer']))
                # q_ids.append(hit['_source']['q_id'])
                data.append(
                    {'question': hit["_source"]['question'], 'answer': hit["_source"]['answer']})
    return data
