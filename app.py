import os
from dotenv import load_dotenv

openai_api_key = os.environ['OPENAI_API_KEY']

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    SystemMessage,
    HumanMessage,
)
speech = """
Ladies and gentlemen,

In our pursuit of knowledge, let us never forget compassion and empathy. Embracing diversity, we find strength. As challenges arise, so does our resilience. Let us shape history with courage and determination, making the world just, equitable, and compassionate for all.

Thank you.
"""
chat_messages = [
    SystemMessage(content="You are an expert assistant with the expertise in summarizing any speech"),
    HumanMessage(content=f"Please provise a summary of the {speech}"),
]
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# speech.get_num_tokens()
print(llm(chat_messages).content)


# Prompt Templates Text Summarization
from langchain.chains import LLMChain
from langchain import PromptTemplate

generic_template = '''
Write a summary of the following speech:
Speech: `{speech}`
Translate the precise summary to {language}
'''
prompt = PromptTemplate(
    input_variables=['speech', 'language'],
    template=generic_template,
)
complete_prompt = prompt.format(speech=speech, language="Hindi")

llm_chain = LLMChain(llm=llm, prompt=prompt)
summary = llm_chain.run({"speech": speech, "language": "Hindi"})
print(summary)



## StuffDocumentChain Text Summarization
from PyPDF2 import PdfReader

# Provide the path to the pdf file
pdfreader = PdfReader("einstein.pdf")

from typing_extensions import Concatenate
# Read text from pdf
text = ""
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        text+=content    
print(text)

from langchain.docstore.document import Document
docs = [Document(page_content=text)]
docs

llm = ChatOpenAI(temperature = 0,model_name="gpt-3.5-turbo")

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document


template = """
Write a concise and short summary of the following speech
Speech : `{text}`.
"""
prompt = PromptTemplate(
    input_variables=["text"],
    template=template
)
chain = load_summarize_chain(
    llm,
    chain_type="stuff",
    prompt= prompt,
    verbose=False
)
output_summary = chain.run(docs)
print(output_summary)


## Summarizing large documents using Map Reduce

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Provide the path of the pdf file
pdfreader = PdfReader("einstein.pdf")
from typing_extensions import Concatenate
# Read text from pdf
text = ""
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        text+=content
        
llm = ChatOpenAI(temperature = 0,model_name="gpt-3.5-turbo")

# Splitting the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
chunks = text_splitter.create_documents({text})

print(chunks)

chain = load_summarize_chain(
    llm,
    chain_type="map_reduce",
    verbose=False
)
summary = chain.run(chunks)
print(summary)


##  Map reduce with custom prompt
chunks_prompt = """
Please summarize the below speech:
text: `{text}`
summary:
"""
map_prompt_template = PromptTemplate(
    input_variables=["text"],
    template=chunks_prompt
)

final_combine_prompt = '''
Provide a final summary of the entire speech with these points.
Add a generic motivational title,
Start the precise summary with an introduction and provide the 
summary in number points for the speech.
Speech: `{text}`
'''
final_combine_prompt_template = PromptTemplate(
    input_variables=["text"],
    template=final_combine_prompt
)

summary_chain = load_summarize_chain(
    llm=llm,
    chain_type="map_reduce",
    map_prompt=map_prompt_template,
    combine_prompt=final_combine_prompt_template,
    verbose=False
)
output = summary_chain.run(chunks)

print(output)

## RefineChain for Summarization
chain = load_summarize_chain(
    llm=llm,
    chain_type="refine",
    verbose=True
)
output_summary = chain.run(chunks)
print(output_summary)