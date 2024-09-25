import streamlit as st
import configparser
from magic import Magic
from PIL import Image
import pdfplumber
from pytesseract import pytesseract 
from docx.api import Document 
from tools import get_current_weather, get_childs_age, get_personal_detail
from langchain_core.messages import AIMessage

from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
# from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain.prompts import PromptTemplate
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from pdf2image import convert_from_bytes, convert_from_path
from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict, List
from langchain_nomic.embeddings import NomicEmbeddings
from io import BytesIO

st.set_page_config(page_title="NA FSI Demo", layout="wide")

st.markdown("""
## NA FSI Demo: Get instant insights from documents which you can upload.

### How It Works

Follow these simple steps to interact with the chatbot:

1. **Upload Your Documents**: The system accepts multiple types of files at once, analyzing the content to provide comprehensive insights.

2. **Enter config properties**: Add different properties to control different aspect of the solution in config/config.properties files.         

4. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
""")



# Load the properties file
config = configparser.ConfigParser()
config.read('config.properties')

def get_config_prop(key, default):
    value = config.get("SETTINGS", key, fallback=default)
    # print("%s --> %s" %(key, value))
    return value

def get_nomic_embedding():
    return NomicEmbeddings(model=get_config_prop("NOMIC_AI_EMBEDDINGS", "nomic-embed-text-v1.5"), inference_mode="local")

def get_llm(model_name = None, tools = False) :
    
    llm = ChatOllama(
        base_url=get_config_prop("OLLAMA_URL","http://localhost:11434"),
        model=model_name or get_config_prop("OLLAMA_MODEL_NAME","llama3.1"), 
        temperature=get_config_prop("TEMPERATURE",0.3)
        )   
   
    if tools:
        llm = llm.bind_tools(
            tools=[get_current_weather, get_childs_age, get_personal_detail],
        )   
        
    return llm

def get_conversational_rag_chain():
    
    prompt = PromptTemplate(
    template="""You are an assistant specializing in question-answering tasks. 
    Your expertise lies in understanding user requests and providing accurate answers based on the DOCUMENTS provided. 
    While you have access to additional tools for further details, use them only when absolutely necessary. 
    If the information is not available, simply state that you do not know.
    USER REQUEST:\n\n {question} \n\n
    DOCUMENTS:\n\n {documents}\n\n
    """,
    input_variables=["question", "documents"],
    ) 

    llm  = get_llm(tools=True)

    rag_chain = prompt | llm     

    return rag_chain

def get_conversational_rag_and_tool_chain():
    

    prompt = PromptTemplate(
    template="""You are an assistant specializing in question-answering tasks. 
    Use the TOOL RESPONSE and DOCUMENTS provided to answer the QUESTION. 
    If the information is not available, simply state that you do not know. 
    QUESTION: {question} 
    TOOL RESPONSE: {tool_output}
    DOCUMENTS: {documents}
    """,
    input_variables=["question", "tool_output", "documents"],
    )

    llm  = get_llm()

    rag_chain = prompt | llm 

    return rag_chain


def get_graph():

    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            search: whether to add search
            documents: list of documents
        """

        question: str
        generation: str
        tool: str
        response: str
        tool_output: str
        documents: List[str]
        steps: List[str]

    def retrieve(state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        question = state["question"]
        documents = get_retriever().invoke(question)
        steps = state["steps"]
        steps.append("retrieve_documents")
        return {"documents": documents, "question": question, "steps": steps}


    def generate(state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """

        question = state["question"]
        documents = state["documents"]
        generation = get_conversational_rag_chain().invoke({"documents": documents, "question": question})
        steps = state["steps"]
        steps.append("generate_answer")

        return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "steps": steps,
        }

    def generate_with_tool_response(state):
        """
        Generate answer based on tool response

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """

        question = state["question"]
        documents = state["documents"]
        tool_output = state["tool_output"]
        generation = get_conversational_rag_and_tool_chain().invoke({"documents": documents, "question": question, "tool_output": tool_output})
        steps = state["steps"]
        steps.append("generate_answer_including_tool_output")

        return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "steps": steps,
        }
    

    def check_for_toolcall(state):
        """
        Determines whether to call tool or not.

        Args:
            state (dict): The current graph state

        Returns:
            str: either return Yes or no to call tool
        """

        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        steps = state["steps"]
        steps.append("check_for_toolcall")

        last_ai_message = None
        
        if isinstance(generation, AIMessage):
            last_ai_message = generation
        
        tool = "No"
        if last_ai_message and hasattr(last_ai_message, 'tool_calls')  and len(last_ai_message.tool_calls) > 0:
            tool = "Yes"
    

        return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "tool": tool,
            "steps": steps,
        }


    def tool_call(state):
        """
        Tool call suggested by llm.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Toll call response
        """

        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
    
        tool_output = ""
        
        tool_mapping = {
        'get_current_weather': get_current_weather,
        'get_childs_age': get_childs_age,
        'get_personal_detail': get_personal_detail
        }
        
        # Extract the last AI message from messages
        last_ai_message = None
        if isinstance(generation, AIMessage):
            last_ai_message = generation
    
        if last_ai_message and hasattr(last_ai_message, 'tool_calls')  and len(last_ai_message.tool_calls) > 0:
            print("last_ai_message:", last_ai_message)
            print("tool name :: ", last_ai_message.tool_calls[-1]["name"])
            print("tool args :: ", last_ai_message.tool_calls[-1]["args"])
            #run tool
            for tool_call in generation.tool_calls:
                tool = tool_mapping[tool_call["name"].lower()]
                tool_output = tool.invoke(tool_call["args"])
        
        steps = state["steps"]
        steps.append("tool_call")

        return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "tool_output": tool_output,
        "steps": steps,
        }
    

    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """
        tool = state["tool"]
        if tool == "Yes":
            return "tool"
        else:
            return "response"


    def generate_response(state):
            
            question = state["question"]
            documents = state["documents"]
            generation = state["generation"]
        
            response = ""

            # Extract the last AI message from messages
            last_ai_message = None
            if isinstance(generation, AIMessage):
                last_ai_message = generation
            else:
                response = generation

            if last_ai_message and hasattr(last_ai_message, 'content') and len(last_ai_message.content.strip()) > 0:
                response = last_ai_message.content
            else:
                response =  generation
            
            steps = state["steps"]
            steps.append("generate_response")

            return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "response": response,
            "steps": steps,
            }


    # Graph
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("check_for_toolcall", check_for_toolcall)  # check_for_toolcall
    workflow.add_node("generate_response", generate_response)  # generate_response
    workflow.add_node("tool_call", tool_call)  # tool_call
    workflow.add_node("generate_with_tool_response", generate_with_tool_response)  # generate_with_tool_response
    
    

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "check_for_toolcall")
    workflow.add_conditional_edges(
            "check_for_toolcall",
            decide_to_generate,
            {
                'tool': "tool_call",
                'response': "generate_response",
            },
        )    

    workflow.add_edge("tool_call", "generate_with_tool_response")
    workflow.add_edge("generate_with_tool_response", "check_for_toolcall")
    workflow.add_edge("generate_response", END)

    custom_graph = workflow.compile()
    
    custom_graph.get_graph(xray=True).draw_mermaid_png()

    graph = custom_graph.get_graph(xray=True).draw_mermaid_png()

    with st.sidebar:
        st.title("Process Flow Diagram")
        st.image(Image.open(BytesIO(graph)), caption="Flow graph")

    return custom_graph



def predict_custom_agent_answer(question):
    
    config = {"configurable": {"thread_id": get_unique_id()}}
    
    state_dict = get_graph().invoke(
        {"question": question, "steps": []}, config
    )

    return {"response": state_dict["response"], "steps": state_dict["steps"]}



def extract_doc_text(doc):
    text = ""
    doc_reader = Document(doc)
    for para in doc_reader.paragraphs:
        text += para.text
    return text



def extract_pdf_text(doc):
    print("Using PDF -> Text for PDF")
    text = ""
    with pdfplumber.open(doc) as pdf_reader:
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_pdf_text_using_pdf2image(doc):
    print("Using PDF -> Image -> Text for PDF")
    text = ""
    # Store Pdf with convert_from_path function
    if isinstance(doc, str):
        images = convert_from_path(doc)    
    else:
        images = convert_from_bytes(doc.getvalue())    
    
    for i in range(len(images)):
    # Save pages as images in the pdf
        if isinstance(doc, str):
            file_path = doc + str(i) +'.jpg'
        else:
            file_path = get_config_prop("ROOT_FOLDER_FOR_DOWNLOAD", "/tmp/") + doc._file_urls.file_id + str(i) +'.jpg'

        images[i].save(file_path, 'JPEG')
        extracted_text = pytesseract.image_to_string(Image.open(file_path))
        text += extracted_text
    
    return text



def get_mime_type(doc):
    if isinstance(doc,st.runtime.uploaded_file_manager.UploadedFile):
        return doc.type
    else:
        mime = Magic(mime=True)
        return mime.from_file(doc)
    
def start_instrument():
    if get_config_prop("INSTRUMENT_TRACE","false") == "true":
        print("Starting tracing...")

        tracer_provider = trace_sdk.TracerProvider()
        trace_api.set_tracer_provider(tracer_provider)
        if get_config_prop("INSTRUMENT_OLTP_TRACE","true") == "true":
            endpoint = get_config_prop("PHOENIX_ENDPOINT","http://127.0.0.1:6006/v1/traces")
            tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
        if get_config_prop("INSTRUMENT_CONSOLE_TRACE","false") == "true":
            tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

        LangChainInstrumentor().instrument()    

def get_unique_id():
    return st.experimental_user.email

def get_retriever():
    new_db = load_chroma_db()
    return new_db.as_retriever(search_kwargs = {"k" : int(get_config_prop("K_VALUE_RETREIVER_COUNT", 1))})
    # return new_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5})

def save_to_chroma_vector_store(text_chunks):
    
    embeddings = get_nomic_embedding()    

    Chroma.from_texts(texts=text_chunks,
                        embedding=embeddings,
                        persist_directory=get_config_prop("CHROMA_DB_DIR_NAME", "db"))    

def load_chroma_db():
    
    embeddings = get_nomic_embedding()
    
    return Chroma(persist_directory=get_config_prop("CHROMA_DB_DIR_NAME", "db"), embedding_function=embeddings)

def user_input(user_question):
    response = predict_custom_agent_answer(user_question)
    
    st.write("Reply: ", response)


def get_extracted_text(docs):
    text = ""
    for doc in docs:

        if get_mime_type(doc) == "application/pdf":
            if st.session_state.pdf_2_image_value == True:
                text += extract_pdf_text_using_pdf2image(doc)
            else:
                text += extract_pdf_text(doc)

        elif get_mime_type(doc) == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text += extract_doc_text(doc)

        else:
            print("Unknown file type")
        
    return text

def get_text_chunks(text):
    chunk_size=int(get_config_prop("CHUNK_SIZE", 1000))
    chunk_overlap=int(get_config_prop("CHUNK_OVERLAP", 100))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks
        

def main():

    start_instrument()

    tab1, tab2 = st.tabs(["Ask Question", "Process Documents"])
    
    with tab1:
        st.header("NA FSI Demo chatbot💁")

        user_question = st.text_input("Ask a Question from the Uploaded files", key="user_question")

        if user_question:  
            user_input(user_question)



    with tab2:
        
        st.title("Process Documents")
        
        if 'pdf_2_image_value' not in st.session_state:
            st.session_state.pdf_2_image_value = False

        st.session_state.pdf_2_image_value = st.checkbox('Use PDF -> Image -> Text!', value=st.session_state['pdf_2_image_value'])

        docs = st.file_uploader("Upload your PDF/DOC files and click on the Submit & Process Button", type=["pdf", "doc", "docx"], accept_multiple_files=True, key="file_uploader")
        if st.button("Submit & Process", key="process_button"):  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                raw_text = get_extracted_text(docs)
                text_chunks = get_text_chunks(raw_text)
                save_to_chroma_vector_store(text_chunks)
                st.success("Done")

            

if __name__ == "__main__":
    main()