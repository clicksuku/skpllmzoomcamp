import csv
import json
import numpy as np

import pandas as pd
from sentence_transformers import SentenceTransformer

from minsearch import VectorSearch, Index
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline



def load_ground_truth():
    records=[]
    with open('ground-truth-data.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            records.append(row)
        return records
    
def load_documents():
    documents = []
    with open('documents-with-ids.json', 'rb') as f_in:
        documents = json.load(f_in)
    return documents

def return_ground_truth():
    records = load_ground_truth()
    df_ground_truth = pd.DataFrame(records)    
    df_ground_truth = df_ground_truth[df_ground_truth[1] == 'machine-learning-zoomcamp']
    ground_truth = df_ground_truth.to_dict(orient='records')
    return ground_truth


def return_embeddings(model, documents):
    vectors = []    
    for doc in documents:
        question = doc['question']
        text = doc['text']
        vector = model.encode(question + ' ' + text)
        vectors.append(vector)
    return vectors

def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)


vindex=None
index=None

def setup_minsearch(is_vector_search:bool):
    global vindex
    global index
    documents=load_documents()
    
    if(not is_vector_search):
        index=Index(text_fields=["question", "section", "text"],
                keyword_fields=["course", "id"])
        index.fit(documents)
    else:
        model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
        vectors = []

        for doc in documents:
            question = doc['question']
            text = doc['text']
            vector = model.encode(question + ' ' + text)
            vectors.append(vector)
        
        vectors = np.array(vectors)
        vindex=VectorSearch(keyword_fields=["course", "id"]) 
        vindex.fit(vectors,documents) 


def setup_minsearch_withSVD(is_vector_search:bool):
    global vindex
    global index
    documents=load_documents()
    
    if(not is_vector_search):
        index=Index(text_fields=["question", "section", "text"],
                keyword_fields=["course", "id"])
        index.fit(documents)
    else:
        vectors = []

        for doc in documents:
            question = doc['question']
            text = doc['text']
            vectors.append(question)
        
        pipeline = make_pipeline(
            TfidfVectorizer(min_df=3),
            TruncatedSVD(n_components=128, random_state=1)
        )
        X = pipeline.fit_transform(vectors)
        vindex=VectorSearch(keyword_fields={"course"}) 
        vindex.fit(X,documents) 
        return vindex, pipeline



def minsearch_text_search(question,course):
    boost = {'question': 1.5, 'section': 0.1}
    return index.search(
        question,
        filter_dict={'course': course},
        boost_dict=boost,
        num_results=5
    )

def minsearch_vector_search(question, course):
    boost = {'question': 1.5, 'section': 0.1}
    model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    v_question = model.encode(question)
    
    return vindex.search(
        v_question,
        filter_dict={'course': course},
        num_results=5
    )

def minsearch_vector_search_withSVD(question, course):
    global vindex
    if(vindex is None):
        vindex,pipeline=setup_minsearch_withSVD(True)

    results = vindex.search(
        pipeline.transform([question])[0],
        num_results=5)
    return results


def compute_relevance(record, result):
    doc_id=record[2]
    relevance=[d['id']==doc_id for d in result]
    return relevance


def search_text(question,course,vector_Search):
    if(vector_Search):
        response=minsearch_vector_search_withSVD(question,course)
        return response
    else:
        response=minsearch_text_search(question,course)
        return response

    

def compute_relevances_forAllQuestions(ground_truth):
    results=[]
    relevances=[]
    for record in ground_truth[:1]:
        question=record[0]
        course=record[1]
        result=search_text(question,course,vector_Search)
        relevance=compute_relevance(record,result)
        print(result)
        print(relevance)
        results.append(result)
        relevances.append(relevance)
    return relevances



if(__name__ == "__main__"):
    vector_Search=True
    #setup_minsearch_withSVD(vector_Search)
    ground_truth=return_ground_truth()
    relevances = compute_relevances_forAllQuestions(ground_truth)
    print("hit_rate:" + str(hit_rate(relevances)))
    print("mrr:" + str(mrr(relevances)))
    
    
    
