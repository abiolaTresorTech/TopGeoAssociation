# import streamlit as st
# import pandas as pd
# import numpy as np

# st.title('Uber pickups in NYC')

import os

import streamlit as st

import streamlit_ext as ste

#from st_files_connection import FilesConnection

from sentence_transformers import SentenceTransformer, util

# import faiss 

#import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

import math as m

# from google.cloud import storage
# from google import cloud
# from google.oauth2 import service_account

# from io import BytesIO

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ds-research-playground-d3bb9f4b9084.json"
# gcs_credentials = {
#     "type": st.secrets["gcs"]["type"],
#     "project_id": st.secrets["gcs"]["project_id"],
#     "private_key_id": st.secrets["gcs"]["private_key_id"],
#     "private_key": st.secrets["gcs"]["private_key"],
#     "client_email": st.secrets["gcs"]["client_email"],
#     "client_id": st.secrets["gcs"]["client_id"],
#     "auth_uri": st.secrets["gcs"]["auth_uri"],
#     "token_uri": st.secrets["gcs"]["token_uri"],
#     "auth_provider_x509_cert_url": st.secrets["gcs"]["auth_provider_x509_cert_url"],
#     "client_x509_cert_url": st.secrets["gcs"]["client_x509_cert_url"],
#     "universe_domain": st.secrets["gcs"]["universe_domain"]
# }


# bucket_name = "datasets-datascience-team"

# list_of_keywords_blob_name =  "datasets-datascience-team/Streamlit_apps_datasets/Keyword_suggestor_datasets/extracted_list_of_keywords_nb_4.pkl"
# index_blob_name = "datasets-datascience-team/Streamlit_apps_datasets/Keyword_suggestor_datasets/extracted_keywords_index_nb_4.bin"

# client = storage.Client()
# credentials = service_account.Credentials.from_service_account_info(gcs_credentials)
# client = storage.Client(credentials=credentials, project=gcs_credentials["project_id"])


# import faiss
# from io import BytesIO

#conn = st.connection('gcs', type=FilesConnection)

# Custom callback reader for reading from bytes
# class BytesReader:
#     def __init__(self, data_bytes):
#         self.data = BytesIO(data_bytes)
    
#     def read(self, size):
#         return self.data.read(size)
    
# def read_gcp_excel_csv_into_pd_df(bucket_name, blob_name, client):
#     bucket = client.get_bucket(bucket_name)
#     blob = bucket.get_blob(blob_name)
#     data_bytes = blob.download_as_bytes()

#     if blob_name.endswith(".csv"):
#         df = pd.read_csv(BytesIO(data_bytes))
#     elif blob_name.endswith(".xlsx"):
#         df = pd.read_csv(BytesIO(data_bytes))
#     elif blob_name.endswith(".pkl"):
#         df = pd.read_pickle(BytesIO(data_bytes))
#     else:
#         raise("The data you're trying to fetch should either be excel file or csv.")
#     return df

# def read_gcp_file_into_bytes(bucket_name, blob_name, client):
#     bucket = client.get_bucket(bucket_name)
#     blob = bucket.get_blob(blob_name)
#     data_bytes = blob.download_as_bytes()
#     return data_bytes


# def read_gcp_bin_into_faiss_index(bucket_name, blob_name, client):
#     ind_as_bytes = read_gcp_file_into_bytes(bucket_name, blob_name, client)

#     # Creating an instance of the custom reader
#     byte_reader = BytesReader(ind_as_bytes)

#     # Creating a FAISS callback IO reader with the custom reader
#     callback_reader = faiss.PyCallbackIOReader(byte_reader.read)

#     # Reading the FAISS index from the callback reader
#     index = faiss.read_index(callback_reader)

#     return index


# st.image("/Users/abiolatresordjigui/DM/streamlit-apps/Data/logo_dm.png", width=100)

st.title("Smart Geography Associator")

if "init" not in st.session_state or not st.session_state.init:
    with st.spinner("Setting everything up..."):
        # index = read_gcp_bin_into_faiss_index(bucket_name, index_blob_name, client)
        # # index = conn.read(bucket_name+"/"+index_blob_name, input_format="binary", ttl=600)
        # st.session_state.index = index
        # extracted_keywords = read_gcp_excel_csv_into_pd_df(bucket_name, list_of_keywords_blob_name, client)["Keyword"].to_list()
        # # extracted_keywords = conn.read(bucket_name+"/"+list_of_keywords_blob_name, input_format="pickle", ttl=600)
        # st.session_state.extracted_keywords = extracted_keywords
        model = SentenceTransformer('all-MiniLM-L6-v2')
        st.session_state.model = model

        st.session_state.init = True
else:
    # index  =st.session_state.index
    # extracted_keywords  =st.session_state.extracted_keywords
    model  =st.session_state.model
#if 'initialized' not in st.session_state or not st.session_state.initialized:
#with st.spinner("Setting things up..."):
    #st.write("ohhhh 1")
    #keywords_embeddings = torch.from_numpy(np.load('/Users/abiolatresordjigui/DM/streamlit-apps/extracted_keywords_normalized_embeddings.npy'))
st.success("Everything was set up successfully!")

with st.expander("Methodology Explanation"):
    st.write("""
    We aim to link different topics to specific geographical areas by evaluating how closely related they are. 
    To do this, we calculate a score that measures the connection between each topic and different regions. 
    This score is based on how likely the topic and region appear together, and it helps us understand which topics 
    are most relevant to which areas. In the end, we generate scores for all regions, and these scores add up to 1, 
    showing the relative importance of each region to the topic.
    """)
#@st.cache_data
def convert_df_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")



# def build_faiss_index(embeddings):
#     d = embeddings.shape[1]
#     index = faiss.IndexFlatL2(d)
#     faiss.normalize_L2(embeddings.numpy())
#     index.add(embeddings.numpy())
#     return index

# def retrieve_documents(query, model,   k=20):
#     query_embedding = model.encode(query, convert_to_tensor=True)
#     faiss.normalize_L2(query_embedding.cpu().numpy().reshape(1, -1))
#     D, I = index.search(query_embedding.cpu().numpy().reshape(1, -1), k)
#     retrieved_docs = [extracted_keywords[i] for i in I[0]]
#     scores = [d for d in D[0]]
#     return retrieved_docs, scores

# def generate_response(retrieved_docs, query):
#     generator = pipeline("text-generation", model="gpt-2")
#     context = " ".join(retrieved_docs)
#     prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
#     response = generator(prompt, max_length=150, num_return_sequences=1)
#     return response[0]['generated_text']


def main():

    # query = st.text_input("Enter a topic here:", placeholder="My Topic")
    uploaded_file_topic_desc = st.file_uploader("Upload topics and descriptions file", type=["xlsx"], help="Upload a one sheet excel file with at least two columns: 'Topic' and 'Description' spelled that same way.")
    uploaded_file_geography = st.file_uploader("Upload geographies file", type=["xlsx"], help="Upload a one sheet excel file with at least one column: 'Geography' spelled that same way.")


    if uploaded_file_topic_desc is not None and uploaded_file_geography is not None:
        df_topic_desc = pd.read_excel(uploaded_file_topic_desc)
        df_geography = pd.read_excel(uploaded_file_geography)
        expected_columns_topic_desc = ["Topic", "Description"]
        expected_columns_geography = ["Geography"]
        columns_are_checked_topic_desc = all([elt in df_topic_desc for elt in expected_columns_topic_desc ])
        columns_are_checked_geography = all([elt in df_geography for elt in expected_columns_geography ])
        if columns_are_checked_topic_desc and columns_are_checked_geography:
            topics2desc = df_topic_desc.groupby("Topic")["Description"].apply(list).to_dict()
            geographies = df_geography["Geography"].to_list()
            output_topic2kw2score = {"Topic":[], "Geography":[], "Score":[]}
            corpus_embeddings = model.encode(geographies,  convert_to_tensor=True)
            # corpus_embeddings = util.normalize_embeddings(corpus_embeddings)
            for topic in topics2desc:
                query = "Let's talk about this topic " + topic +". " + topics2desc[topic][0]
                query_embeddings = model.encode([query], convert_to_tensor=True)
                # query_embeddings = util.normalize_embeddings(query_embeddings)
                hits_for_stats = util.semantic_search(query_embeddings, corpus_embeddings, top_k = len(geographies), score_function = util.dot_score)[0]
                topics = [topic for _ in hits_for_stats]
                scored_geographies = [geographies[hit["corpus_id"]] for hit in hits_for_stats]
                scores = [hit["score"] for hit in hits_for_stats]
                scores = [m.exp(score) for score in scores]
                scores = [m/sum(scores) for m in scores]
                output_topic2kw2score["Topic"].extend(topics)
                output_topic2kw2score["Geography"].extend(scored_geographies)
                output_topic2kw2score["Score"].extend(scores)
                
            output_topic2kw2score_df = pd.DataFrame.from_dict(output_topic2kw2score)
            output_topic2kw2score_csv = convert_df_to_csv(output_topic2kw2score_df)

            heatmap_data = output_topic2kw2score_df.pivot(index='Geography', columns='Topic', values='Score')
            plt.figure(figsize=(len(geographies)/3 , len(topics2desc)))
            heatmap = sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', annot_kws={"size": 5})
            # plt.tight_layout()
            plt.xticks(fontsize=5)  # Set x-axis (Topic) label font size
            plt.yticks(fontsize=5) 

            plt.xlabel('Topic', fontsize=8)  # Set x-axis title size
            plt.ylabel('Geography', fontsize=8)
            cbar = heatmap.collections[0].colorbar
            cbar.ax.tick_params(labelsize=5)  
            st.pyplot(plt)
            # st.write("yooooo")
            # plt.clf()
            ste.download_button(
            label="Download results as CSV",
            data=output_topic2kw2score_csv,
            file_name="topic_geographies.csv",
            mime="text/csv",
        )
        else:
            st.error('One of your csv files has wrong formatting. \
                     In the topic description file, make sure you have these two columns spelled that same way: {}. \
                     In the geographies file, make sure you have these two columns spelled that same way: {}'.format(expected_columns_topic_desc, expected_columns_geography), icon="🚨")


    # if query:
    #     retrieved_docs, scores = retrieve_documents(query, model)
    #     result_data = pd.DataFrame({"Keyword":retrieved_docs, "Score":scores})

    #     st.write("Response")
    #     st.dataframe(result_data, use_container_width = True)

    #     result_data_as_csv = convert_df_to_csv(result_data)
    #     ste.download_button(
    #     label="Download results as CSV",
    #     data=result_data_as_csv,
    #     file_name="suggested_keywords_for_{}.csv".format(query),
    #     mime="text/csv",
    # )

if __name__ == "__main__":
    main()
