from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

class VectorStore:
    def __init__(self, persist_directory="./chroma_db"):
        """Initialize with Hugging Face embeddings"""
        #using local embedding model (reduce cost, I am broke)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  
        )
        
        self.persist_directory = persist_directory
        self.vectorstore = None
    
    def create_vectorstore(self, chunks):
        """Convert document chunks to embeddings and store in ChromaDB"""
        
        print(f"Creating embeddings for {len(chunks) } chunks...")
        
        #embedding call
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        print(f"Vectorstore created and saved to {self.persist_directory}")
        return self.vectorstore

    def load_vectorstore(self):
        """Load existing vectorstore from disk"""
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        print("Loaded existing vectorstore")
        return self.vectorstore

    def search(self, query, k=3, filter_dict=None):
        """Find most relevant chunks for a question"""
        if not self.vectorstore:
            raise ValueError("No vectorstore loaded. Call create_vectorstore() or load_vectorstore() first")
        
        #find similar chunks
        results = self.vectorstore.similarity_search(
            query, 
            k=k,  # Number of chunks to return
            filter=filter_dict  # e.g., {"source_type": "course_material"}
        )
        
        print(f"Found {len(results)} relevant chunks for: '{query}'")
        return results