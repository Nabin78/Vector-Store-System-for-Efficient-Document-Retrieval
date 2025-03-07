from fastapi import APIRouter, HTTPException
from app.routers.text.new_model import LlamaIndexFAISSManager, VectorStoreResponse, AddVectorResponse, AddVectorsRequest, SearchVectorsRequest, DeleteVectorRequest, UpdateVectorRequest, UpdateVectorStoreRequest, ComputeSimilarityRequest

router = APIRouter(tags=["Text"])

vector_manager = LlamaIndexFAISSManager()


    
@router.post("/create/{name}", response_model=VectorStoreResponse)
async def create_vector_store(name: str):
    """Creates a vector store associated with a unique token using LlamaIndex integration."""
    
    if name.strip() == "":
        raise HTTPException(status_code=400, detail="Name is required.")
    
    try:
        token = vector_manager.create_by_name(name)
        return VectorStoreResponse(
            auth_token=token,
            message=f"Vector store '{name}' created with LlamaIndex integration."
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error creating vector store: {str(e)}"
        )

@router.post("/add_vector", response_model=AddVectorResponse)
async def add_vectors(request: AddVectorsRequest):
    """
    Adds vectors to the store using LlamaIndex document processing.
    
    The text will be processed and embedded using LlamaIndex's document processing
    pipeline before being added to the vector store.
    """
    
    if not request.auth_token or not request.texts or not request.subject:
        raise HTTPException(
            status_code=400,
            detail="auth_token, subject, and texts are required fields."
        )

    try:
        metadata = {
            "subject": request.subject,
            **(request.metadata or {})
        }
        
        vector_id = vector_manager.add_vector_by_token_and_text(
            request.auth_token,
            request.texts,
            metadata=metadata
        )
        
        return AddVectorResponse(
            message="Vector added successfully using LlamaIndex integration.",
            subject=request.subject,
            vector_id = vector_id
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error adding vector: {str(e)}"
        )

@router.post("/search/{k}")
async def search_k_vectors(
    k: int,
    request: SearchVectorsRequest
    ):
    """
    Searches for the top k similar vectors using LlamaIndex's retriever query engine.
    
    This endpoint leverages LlamaIndex's advanced retrieval capabilities including
    semantic search and relevancy scoring.
    """
    
    if not request.auth_token or not request.input_query_context.strip():
        raise HTTPException(
            status_code=400,
            detail="auth_token and input_query_context are required fields."
        )

    if k < 1:
        raise HTTPException(
            status_code=400,
            detail="k must be greater than 0"
        )

    try:
        results = vector_manager.search_vector_by_token_and_k(
            request.auth_token,
            request.input_query_context,
            k
        )
        
        return results
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error performing search: {str(e)}"
        )





# Endpoints

@router.put("/update_vector_store")
async def update_vector_store(request: UpdateVectorStoreRequest):
    """Updates the name of an existing vector store."""
    try:
        vector_manager.update_vector_store(request.auth_token, request.new_name)
        return {"message": f"Vector store updated successfully to '{request.new_name}'."}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating vector store: {str(e)}"
        )


@router.delete("/delete_vector_store/{auth_token}")
async def delete_vector_store(auth_token: str):
    """Deletes a vector store and its associated data."""
    try:
        vector_manager.delete_vector_store(auth_token)
        return {"message": "Vector store deleted successfully."}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting vector store: {str(e)}"
        )


@router.put("/update_vector")
async def update_vector(request: UpdateVectorRequest):
    """
    Updates an existing vector based on similarity checks and user input.
    Requires the vector's unique ID (`vector_id`) and updated text.
    """
    try:
        vector_manager.update_vectors(request.auth_token, request.vector_id, request.new_text, request.action)
        return {"message": f"Vector with ID '{request.vector_id}' updated successfully."}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating vector: {str(e)}"
        )


@router.delete("/delete_vector")
async def delete_vector(request: DeleteVectorRequest):
    """
    Deletes an individual vector from a vector store using its unique ID (`vector_id`).
    """
    try:
        vector_manager.delete_vector(request.auth_token, request.vector_id)
        return {"message": f"Vector with ID '{request.vector_id}' deleted successfully."}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting vector: {str(e)}"
        )

@router.post("/compute_similarity")
async def compute_similarity(request: ComputeSimilarityRequest):
    """
    Compute the cosine similarity between an existing vector and a new text.

    Args:
        auth_token (str): The authentication token for the vector store.
        vector_id (str): The ID of the vector to compare.
        new_text (str): The new text for comparison.

    Returns:
        dict: A JSON response containing the similarity score.
    """
    try:
        similarity = vector_manager.compute_cosine_similarity(request.auth_token, request.vector_id, request.new_text)
        return {"similarity": similarity}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing similarity: {str(e)}")
