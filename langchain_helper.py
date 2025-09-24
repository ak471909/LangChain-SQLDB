from langchain.llms import google_palm 
from langchain.utilities import SQLDatabase 
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt 
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI, OpenAIEmbeddings
import os
from few_shots import few_shots

from dotenv import load_dotenv
load_dotenv()

def get_few_shot_db_chain():

  llm = OpenAI(
          openai_api_key=os.environ["OPENAI_API_KEY"],
          temperature=0.1,
      )

  db_user = "root"
  db_password = "root"
  db_host = "localhost"
  db_name = "atliq_tshirts"

  db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

  embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
  to_vectorize = [" ".join(example.values()) for example in few_shots]
  vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)
  example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2
  )

  example_prompt = PromptTemplate(
    input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
    template = "\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}\n")

  few_shot_prompt = FewShotPromptTemplate(
    example_selector = example_selector, 
    example_prompt=example_prompt, 
    prefix=_mysql_prompt, 
    suffix=PROMPT_SUFFIX,
    input_variables=["input", "table_info","top_k"]
  )

  chain= SQLDatabaseChain.from_llm(
    llm,
    db,
    prompt=few_shot_prompt,
    verbose=True,
    return_intermediate_steps=True,
    top_k=2
  )
  return chain

if __name__ == "__main__":
  chain = get_few_shot_db_chain()
  print(chain.run("How many total t shirts are left in stock"))