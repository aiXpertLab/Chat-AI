import os

from langchain.chains import APIChain
from langchain.chat_models import ChatOpenAI
from utils.LangChain_Routine import llm
from utils.LangChain_Prompt  import GeneralPromptTemplate

# 1. Load the LlamaCpp language model, adjust GPU usage based on your hardware
llm = llm()

API_DOCS = """API documentation:
Endpoint: https://api.themoviedb.org/3
GET /search/movie

This API is for searching movies.

Query parameters table:
language | string | Pass a ISO 639-1 value to display translated data for the fields that support it. minLength: 2, pattern: ([a-z]{2})-([A-Z]{2}), default: en-US | optional
query | string | Pass a text query to search. This value should be URI encoded. minLength: 1 | required
page | integer | Specify which page to query. minimum: 1, maximum: 1000, default: 1 | optional
include_adult | boolean | Choose whether to include adult (pornography) content in the results. default | optional
region | string | Specify a ISO 3166-1 code to filter release dates. Must be uppercase. pattern: ^[A-Z]{2}$ | optional
year | integer  | optional
primary_release_year | integer | optional

Response schema (JSON object):
page | integer | optional
total_results | integer | optional
total_pages | integer | optional
results | array[object] (Movie List Result Object)

Each object in the "results" key has the following schema:
poster_path | string or null | optional
adult | boolean | optional
overview | string | optional
release_date | string | optional
genre_ids | array[integer] | optional
id | integer | optional
original_title | string | optional
original_language | string | optional
title | string | optional
backdrop_path | string or null | optional
popularity | number | optional
vote_count | integer | optional
video | boolean | optional
vote_average | number | optional"""

def settings():

 callbacks = [StreamingStdOutCallbackHandler()]

    # LLM (Place your model into models directory).
    llm = GPT4All(model="models/WizardLM-7B-uncensored.ggmlv3.q8_0.bin", max_tokens=800, backend='gptj' ,callbacks=callbacks,verbose=false)

    access_token = "YOUR_API_ACCESS_TOKEN" #Retrieve the access token for your APIs dynamically.
    headers = {"Authorization" : f"Bearer {access_token}" }
    
    #Create an APIChain to interpret the input prompt using API DOCS and LLM.
    chain = APIChain.from_llm_and_api_docs(llm, API_DOCS, headers=headers, verbose=True)

    return chain

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.info(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # self.container.write(documents)
        for idx, doc in enumerate(documents):
            source = doc.metadata["source"]
            self.container.write(f"**Results from {source}**")
            self.container.text(doc.page_content)


    st.header("`Interacting with APIs`")

    # Make APIChain 
    if 'chain' not in st.session_state:
        st.session_state['chain'] = settings()
    chain = st.session_state.chain

    # User input 
    question = st.text_input("`Ask a question:`")
    
    if question:
    
        # Generate answer (Make an API call with input from prompt used to 
        # create request payload or request params or query params or 
        # path params

        qa_chain = chain.run(question)
    
        # Write answer and sources
        retrieval_streamer_cb = PrintRetrievalHandler(st.container())
        answer = st.empty()
        answer.info('`Answer:`\n\n' + qa_chain)
        
        
