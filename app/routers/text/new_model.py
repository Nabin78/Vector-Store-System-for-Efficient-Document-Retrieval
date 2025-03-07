from pydantic import BaseModel
from typing import List, Optional, Dict
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import Document, QueryBundle
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import faiss
import traceback
import uuid
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity #use to extract text
import os
import json
from datetime import datetime
import hashlib #imp

class VectorStoreResponse(BaseModel):
    auth_token: str
    message: str

class AddVectorResponse(BaseModel):
    message: str
    subject: str
    vector_id: Optional[str]
    
class AddVectorsRequest(BaseModel):
    auth_token: str
    texts: str
    subject: Optional[str]
    metadata: Optional[dict] = {}

class SearchVectorsRequest(BaseModel):
    auth_token: str
    input_query_context: str

# Request models for new endpoints
class UpdateVectorStoreRequest(BaseModel):
    auth_token: str
    new_name: str

class UpdateVectorRequest(BaseModel):
    auth_token: str
    vector_id: str
    new_text: str
    action: str

class DeleteVectorRequest(BaseModel):
    auth_token: str
    vector_id: str

class ComputeSimilarityRequest(BaseModel):
    auth_token: str
    vector_id: str
    new_text: str

class LlamaIndexFAISSManager:
    def __init__(self, token_file="token_list.json", vector_store_dir="vector_stores"): 
        
        self.token_file = token_file
        self.vector_store_dir = vector_store_dir
        self.vector_stores: Dict[str, FaissVectorStore] = {}
        self.indexes: Dict[str, VectorStoreIndex] = {}
        self.metadata = {}
        
        if not os.path.exists(vector_store_dir):
            os.makedirs(vector_store_dir)
        if not os.path.exists(token_file):
            with open(token_file, "w") as f:
                json.dump({}, f)

        Settings.llm = OpenAI(temperature=0.0, model="gpt-4o-mini")
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        # Settings.chunk_size = 1024
        # Settings.chunk_overlap = 20

    def create_by_name(self, name: str) -> str:
        """Creates a new FAISS vector store with LlamaIndex integration."""
        
        token = hashlib.md5(name.encode()).hexdigest()

        self._check_name_exists(name)

        dimension = 1536  #imp
        faiss_index = faiss.IndexFlatL2(dimension) #imp
        
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        index = VectorStoreIndex([], storage_context=storage_context)
        
        self.indexes[token] = index
        self.vector_stores[token] = vector_store
        self.metadata[token] = {
            "name": name,
            "created_on": datetime.now().strftime("%Y-%m-%d-%H-%M"),
            "vectors": [],
            "faiss_to_vector_id": {},  # Initialize FAISS ID to vector ID mapping
            "vector_id_to_faiss": {}   # Initialize vector ID to FAISS ID mapping
        }
        
        print(f"Creating vector store - Name: {name}, Token: {token}")
        self.save_vector_store(token)
        self._save_token(name, token)
        
        return token

    

    # def add_vector_by_token_and_text(self, auth_token: str, input_text: str, metadata: Optional[Dict] = None) -> str:
    #     self._check_token_exists(auth_token)
    #     print(f"\nAdding document - Token: {auth_token}")

    #     if auth_token not in self.vector_stores:
    #         self.load_vector_store(auth_token)

    #     # Generate unique hash ID
    #     unique_vector_id = hashlib.md5(f"{input_text}-{uuid.uuid4()}".encode()).hexdigest()

    #     # Generate numeric ID for FAISS
    #     faiss_id = len(self.metadata[auth_token].get("vector_map", {}))

    #     # Add mapping to metadata
    #     self.metadata[auth_token].setdefault("vector_map", {})[unique_vector_id] = faiss_id

    #     # Create the document
    #     doc = Document(
    #         text=input_text,
    #         id_=str(faiss_id),
    #         metadata={**(metadata or {}), "vector_id": unique_vector_id},
    #     )

    #     vector_store = self.vector_stores[auth_token]
    #     embedding = Settings.embed_model.get_text_embedding(input_text)
    #     vector_store._faiss_index.add(np.array([embedding]))

    #     # Add to nodes_dict
    #     self.indexes[auth_token].index_struct.nodes_dict[str(faiss_id)] = {
    #         "text": input_text,
    #         "metadata": doc.metadata,
    #     }

    #     # Save metadata
    #     now = datetime.now().strftime("%Y-%m-%d-%H-%M")
    #     self.metadata[auth_token]["vectors"].append({
    #         "input_text": input_text,
    #         "added_on": now,
    #         "vector_id": unique_vector_id,
    #     })

    #     self.save_vector_store(auth_token)
    def add_vector_by_token_and_text(self, auth_token: str, input_text: str, metadata: Optional[Dict] = None) -> str:
        self._check_token_exists(auth_token)
        print(f"Adding document - Token: {auth_token}")

        if auth_token not in self.vector_stores:
            self.load_vector_store(auth_token)

        # Generate a unique string-based vector ID
        vector_id = hashlib.md5(f"{input_text}-{uuid.uuid4()}".encode()).hexdigest()
        faiss_id = self.vector_stores[auth_token]._faiss_index.ntotal 

        #Add mappings
        self.metadata[auth_token]["faiss_to_vector_id"][faiss_id] = vector_id
        self.metadata[auth_token]["vector_id_to_faiss"][vector_id] = faiss_id

        # Generate embeddings for the input text
        embedding = Settings.embed_model.get_text_embedding(input_text)
        embedding_np = np.array(embedding).reshape(1, -1)

        # Add to FAISS and get the assigned integer ID
        vector_store = self.vector_stores[auth_token]
        faiss_id = vector_store._faiss_index.ntotal  # Current FAISS ID
        vector_store._faiss_index.add(embedding_np)

        # Store the mapping between FAISS ID and vector ID
        self.metadata[auth_token]["faiss_to_vector_id"][faiss_id] = vector_id
        self.metadata[auth_token]["vector_id_to_faiss"][vector_id] = faiss_id

        # Store the vector metadata
        now = datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.metadata[auth_token]["vectors"].append({
            "vector_id": vector_id,
            "input_text": input_text,
            "added_on": now,
        })

        self.save_vector_store(auth_token)
        print(f"Successfully added document - Vector ID: {vector_id}, FAISS ID: {faiss_id}")
        return vector_id

        




    # def search_vector_by_token_and_k(self, auth_token: str, search_context: str, k: int) -> List[Dict]:
    #     """Searches vectors using LlamaIndex's retriever query engine and returns formatted results."""
        
    #     self._check_token_exists(auth_token)
    #     print(f"\nSearching vectors - Token: {auth_token}")
        
    #     if auth_token not in self.vector_stores:
    #         self.load_vector_store(auth_token)
        
    #     index = self.indexes[auth_token]
        
    #     try:
    #         retriever = VectorIndexRetriever(
    #             index=index,
    #             similarity_top_k=min(k, len(self.metadata[auth_token]["vectors"]))
    #         )
            
    #         query_bundle = QueryBundle(query_str=search_context)
    #         retrieved_nodes = retriever.retrieve(query_bundle)
            
    #         results = []
    #         for node in retrieved_nodes:
    #             if not hasattr(node, 'node') or not hasattr(node.node, 'text'):
    #                 continue
                    
    #             node_metadata = node.node.metadata if hasattr(node.node, "metadata") else {}
                
    #             result = {
    #                 "matched_text": node.node.text,
    #                 "score": float(node.score) if hasattr(node, 'score') else None
    #             }
                
    #             if "subject" in node_metadata:
    #                 result["subject"] = node_metadata["subject"]
                    
    #             results.append(result)
            
    #         return results
                
    #     except Exception as e:
    #         print(f"Error during search: {str(e)}")
    #         traceback.print_exc()
    #         raise ValueError(f"Error performing search: {str(e)}")
    def search_vector_by_token_and_k(self, auth_token: str, search_context: str, k: int) -> List[Dict]:
        self._check_token_exists(auth_token)
        print(f"\nSearching vectors - Token: {auth_token}")

        if auth_token not in self.vector_stores:
            self.load_vector_store(auth_token)

        index = self.indexes[auth_token]

        try:
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=min(k, len(self.metadata[auth_token]["vectors"]))
            )
            query_bundle = QueryBundle(query_str=search_context)
            query_result = retriever.retrieve(query_bundle)

            # Map FAISS IDs to vector IDs
            results = []
            for idx in query_result.ids:
                faiss_id = str(idx)
                if faiss_id not in self.metadata[auth_token]["faiss_to_vector_id"]:
                    print(f"Error: FAISS ID {faiss_id} missing in faiss_to_vector_id.")
                    continue  # Skip missing mappings

                vector_id = self.metadata[auth_token]["faiss_to_vector_id"][faiss_id]
                
                node_data = self.indexes[auth_token].index_struct.nodes_dict[vector_id]

                results.append({
                    "vector_id": vector_id,
                    "matched_text": node_data["text"],
                    "metadata": node_data["metadata"]
                })

            return results

        except Exception as e:
            print(f"Error during search: {e}")
            raise ValueError(f"Error performing search: {e}")
    # def search_vector_by_token_and_k(self, auth_token: str, search_context: str, k: int) -> List[Dict]:
    #     """Searches vectors using LlamaIndex's retriever query engine and returns formatted results."""
        
    #     self._check_token_exists(auth_token)
    #     print(f"\nSearching vectors - Token: {auth_token}")
        
    #     if auth_token not in self.vector_stores:
    #         self.load_vector_store(auth_token)
        
    #     index = self.indexes[auth_token]
        
    #     try:
    #         retriever = VectorIndexRetriever(
    #             index=index,
    #             similarity_top_k=min(k, len(self.metadata[auth_token]["vectors"]))
    #         )
            
    #         query_bundle = QueryBundle(query_str=search_context)
    #         retrieved_nodes = retriever.retrieve(query_bundle)
            
    #         results = []
    #         for node in retrieved_nodes:
    #             if not hasattr(node, 'node') or not hasattr(node.node, 'text'):
    #                 continue
                    
    #             node_metadata = node.node.metadata if hasattr(node.node, "metadata") else {}
                
    #             result = {
    #                 "matched_text": node.node.text,
    #                 "score": float(node.score) if hasattr(node, 'score') else None
    #             }
                
    #             if "subject" in node_metadata:
    #                 result["subject"] = node_metadata["subject"]
                    
    #             results.append(result)
            
    #         return results
                
    #     except Exception as e:
    #         print(f"Error during search: {str(e)}")
    #         traceback.print_exc()
    #         raise ValueError(f"Error performing search: {str(e)}")


    
    def save_vector_store(self, auth_token: str) -> None:
        """Saves the vector store and metadata to disk."""
        if auth_token not in self.vector_stores:
            raise ValueError(f"No vector store found for token '{auth_token}'")
        
        vector_store = self.vector_stores[auth_token]
        if not hasattr(vector_store, '_faiss_index'):
            raise ValueError("Vector store does not contain a FAISS index")
        
        faiss_path = os.path.join(self.vector_store_dir, f"{auth_token}.index")
        faiss.write_index(vector_store._faiss_index, faiss_path)
        
        metadata_path = os.path.join(self.vector_store_dir, f"{auth_token}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata[auth_token], f)
        
        print(f"Saved vector store - Token: {auth_token}")
        print(f"Current vector count: {vector_store._faiss_index.ntotal}")

    def load_vector_store(self, auth_token: str) -> None:
        """Loads the vector store and metadata from disk."""
        self._check_token_exists(auth_token)
        print(f"\nLoading vector store - Token: {auth_token}")
        
        metadata_path = os.path.join(self.vector_store_dir, f"{auth_token}_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata[auth_token] = json.load(f)
        else:
            self.metadata[auth_token] = {"vectors": []}

        # Ensure faiss_to_vector_id and vector_id_to_faiss mappings exist in metadata
        if "faiss_to_vector_id" not in self.metadata[auth_token]:
            self.metadata[auth_token]["faiss_to_vector_id"] = {}
        if "vector_id_to_faiss" not in self.metadata[auth_token]:
            self.metadata[auth_token]["vector_id_to_faiss"] = {}

        faiss_path = os.path.join(self.vector_store_dir, f"{auth_token}.index")
        if not os.path.exists(faiss_path):
            raise ValueError(f"FAISS store not found for token '{auth_token}'")
        
        faiss_index = faiss.read_index(faiss_path)
        print(f"Loaded FAISS index - Vector count: {faiss_index.ntotal}")
        
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        
        documents = []
        for idx, vector_info in enumerate(self.metadata[auth_token]["vectors"]):
            vector_id = vector_info.get("vector_id", str(idx))
            self.metadata[auth_token]["faiss_to_vector_id"][idx] = vector_id
            self.metadata[auth_token]["vector_id_to_faiss"][vector_id] = idx

            doc = Document(
                text=vector_info["input_text"],
                id_=str(idx), 
                metadata={"vector_id": idx}
            )
            documents.append(doc)
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        self.vector_stores[auth_token] = vector_store
        self.indexes[auth_token] = index
            
        print(f"Successfully loaded vector store - Documents: {len(documents)}")

    def _check_name_exists(self, name: str) -> None:
        with open(self.token_file, "r") as f:
            token_data = json.load(f)
        if name in token_data:
            raise ValueError(f"Vector store with name '{name}' already exists")

    def _save_token(self, name: str, token: str) -> None:
        with open(self.token_file, "r") as f:
            token_data = json.load(f)
        token_data[name] = token
        with open(self.token_file, "w") as f:
            json.dump(token_data, f)

    def _check_token_exists(self, auth_token: str) -> None:
        with open(self.token_file, "r") as f:
            token_data = json.load(f)
        if auth_token not in token_data.values():
            raise ValueError("Invalid token, no vector store found")
        
    def update_vector_store(self, auth_token: str, new_name: str) -> None:
        """Updates the name of a vector store."""
        self._check_token_exists(auth_token)
        old_name = None

        with open(self.token_file, "r") as f:
            token_data = json.load(f)

        # Find the old name associated with the token
        for name, token in token_data.items():
            if token == auth_token:
                old_name = name
                break

        if old_name is None:
            raise ValueError("Vector store name not found.")

        # Update token data with new name
        token_data.pop(old_name)
        token_data[new_name] = auth_token

        with open(self.token_file, "w") as f:
            json.dump(token_data, f)

        self.metadata[auth_token]["name"] = new_name
        print(f"Updated vector store name from '{old_name}' to '{new_name}'.")

    def delete_vector_store(self, auth_token: str) -> None:
        """Deletes a vector store and its associated files."""
        self._check_token_exists(auth_token)

        # Paths for metadata and FAISS index
        metadata_path = os.path.join(self.vector_store_dir, f"{auth_token}_metadata.json")
        faiss_path = os.path.join(self.vector_store_dir, f"{auth_token}.index")

        # Remove metadata and FAISS files
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        if os.path.exists(faiss_path):
            os.remove(faiss_path)

        # Remove the token from token list
        with open(self.token_file, "r") as f:
            token_data = json.load(f)

        token_data = {k: v for k, v in token_data.items() if v != auth_token}

        with open(self.token_file, "w") as f:
            json.dump(token_data, f)

        # Clean up memory
        self.vector_stores.pop(auth_token, None)
        self.indexes.pop(auth_token, None)
        self.metadata.pop(auth_token, None)

        print(f"Deleted vector store with token: {auth_token}.")

    
    def update_vectors(self, auth_token: str, vector_id: str, new_text: str, action: str) -> None:
        """
        Updates an existing vector or adds a new vector based on the user's choice.
        
        Args:
            auth_token (str): Authentication token for the vector store.
            vector_id (str): ID of the vector to update.
            new_text (str): New text to update or add as a new vector.
            action (str): User's choice - 'Cancel', 'Replace', or 'Add as New'.
        """
        # Ensure the vector store is loaded
        if auth_token not in self.vector_stores:
            print(f"Loading vector store for auth_token: {auth_token}")
            try:
                self.load_vector_store(auth_token)
            except Exception as e:
                raise ValueError(f"Vector store not found or could not be loaded: {str(e)}")

        if action == "Cancel":
            print("User chose to leave the vector unchanged.")
            return

        elif action == "Replace":
            # Retrieve metadata for the vector
            vectors_metadata = self.metadata.get(auth_token, {}).get("vectors", [])
            existing_vector = next((v for v in vectors_metadata if v["vector_id"] == vector_id), None)

            if not existing_vector:
                raise ValueError(f"No vector found with ID {vector_id}.")

            # Update the vector's text
            existing_vector["input_text"] = new_text
            print("Updated the vector with new text.")

        elif action == "Add as New":
            # Add the new text as a separate vector
            self.add_vector_by_token_and_text(auth_token, new_text)
            print("Added the new vector.")
        
        else:
            raise ValueError("Invalid action. Must be 'Cancel', 'Replace', or 'Add as New'.")

        # Save the updated vector store
        self.save_vector_store(auth_token)


    # def delete_vector(self, auth_token: str, vector_id: int) -> None:
    #     """Deletes a vector based on its vector ID."""
    #     self._check_token_exists(auth_token)
    #     if auth_token not in self.vector_stores:
    #         self.load_vector_store(auth_token)

    #     vectors = self.metadata[auth_token]["vectors"]
    #     vector_index = next((i for i, v in enumerate(vectors) if v["vector_id"] == vector_id), None)

    #     if vector_index is None:
    #         raise ValueError(f"No vector found with ID {vector_id}.")

    #     del vectors[vector_index]
    #     print(f"Deleted vector with ID {vector_id}.")

    #     self.save_vector_store(auth_token)

    def delete_vector(self, auth_token: str, vector_id: str) -> None:
        self._check_token_exists(auth_token)
        if auth_token not in self.vector_stores:
            self.load_vector_store(auth_token)

        vector_store = self.vector_stores[auth_token]
        faiss_index = vector_store._faiss_index

        if vector_id not in self.metadata[auth_token]["vector_id_to_faiss"]:
            raise ValueError(f"No vector found with ID {vector_id}.")

        faiss_id = self.metadata[auth_token]["vector_id_to_faiss"].pop(vector_id)
        self.metadata[auth_token]["faiss_to_vector_id"].pop(faiss_id)

        # Remove vector from FAISS
        faiss_index.remove_ids(np.array([faiss_id]))

        # Remove vector metadata
        self.metadata[auth_token]["vectors"] = [
            v for v in self.metadata[auth_token]["vectors"] if v["vector_id"] != vector_id
        ]

        self.save_vector_store(auth_token)
        print(f"Vector {vector_id} removed successfully.")



    def compute_cosine_similarity(self, auth_token: str, vector_id: str, new_text: str) -> float:
        """
        Compute the cosine similarity between an existing vector and a new text.

        Args:
            auth_token (str): Authentication token for the vector store.
            vector_id (str): ID of the vector to compare.
            new_text (str): New text for comparison.

        Returns:
            float: The cosine similarity score between the existing and new text.
        """

        self.load_vector_store(auth_token)

        # Validate vector store existence
        if auth_token not in self.vector_stores:
            raise ValueError("Vector store not found for the provided auth_token.")
            

        # Retrieve metadata for the vector
        vectors_metadata = self.metadata[auth_token]["vectors"]
        existing_vector = next((v for v in vectors_metadata if v["vector_id"] == vector_id), None)

        if not existing_vector:
            raise ValueError("Vector ID not found in the vector store.")

        # Fetch existing vector text
        existing_text = existing_vector["input_text"]

        # Generate embeddings for the existing and new text
        embed_model = Settings.embed_model  # Assuming embedding model is initialized in Settings
        existing_embedding = np.array(embed_model.get_text_embedding(existing_text))
        new_embedding = np.array(embed_model.get_text_embedding(new_text))

        # Compute cosine similarity
        similarity = cosine_similarity([existing_embedding], [new_embedding])[0][0]

        return similarity

