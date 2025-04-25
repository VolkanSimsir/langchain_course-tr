from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


class RagBot:
    def __init__(self,temperature,model_name, file_path):
        self.temperature = temperature
        self.model_name = model_name

        self.load_model()
        self.split_file(file_path=file_path)

        self.vectorstore = None
        if self.vectorstore is None:
            self.load_embeddings()

    def load_model(self):
        model = ChatOpenAI(model_name=self.model_name,temperature=self.temperature)
        self.model = model
        
    def load_file(self,file_path):
        loader = PyPDFLoader(file_path=file_path)
        docs = loader.load()
        return docs

    def split_file(self,file_path):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = self.load_file(file_path)
        split_docs = text_splitter.split_documents(docs)
        self.split_docs = split_docs
        

    def load_embeddings(self):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(self.split_docs, embeddings)
        self.vectorstore = vectorstore
        

    def generate_response(self,user_query):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.model,
            retriever=retriever,
            memory=memory
        )
        result = qa_chain.invoke({"question": user_query})
        return result["answer"]
        

if __name__ == "__main__":
    bot = RagBot(temperature=0.7,model_name="gpt-3.5-turbo",file_path="")
    response = bot.generate_response(user_query="what is the this paper title name")
    print(response)
