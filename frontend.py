import streamlit as st
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Base URL for the FastAPI backend
BASE_URL = "http://127.0.0.1:8000"  # Replace with your actual FastAPI server URL

st.title("Vector Store Management UI")

import openai
import numpy as np

# Set up your OpenAI API key
openai.api_key = "your-openai-api-key"

def get_text_embedding(text: str):
    """
    Generates an embedding for the given text using OpenAI's embedding API.

    Args:
        text (str): The input text.

    Returns:
        np.ndarray: The embedding as a NumPy array.
    """
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"  # Use OpenAI's embedding model
    )
    embedding = response["data"][0]["embedding"]
    return np.array(embedding)


# Tabbed interface for managing vector stores and vectors
tabs = st.tabs([
    "Create Vector Store",
    "Delete Vector Store",
    "Update Vector Store",
    "Add Vector",
    "Search Vectors",
    "Update Vector",
    "Delete Vector"
])

# ----------- Tab 1: Create Vector Store -----------
with tabs[0]:
    st.header("Create Vector Store")
    vector_store_name = st.text_input("Enter a name for the vector store:")
    if st.button("Create Vector Store"):
        if vector_store_name.strip():
            response = requests.post(f"{BASE_URL}/create/{vector_store_name}")
            if response.status_code == 200:
                st.success(f"Vector Store Created: {response.json()['auth_token']}")
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown Error')}")
        else:
            st.error("Please provide a name for the vector store.")

# ----------- Tab 2: Delete Vector Store -----------
with tabs[1]:
    st.header("Delete Vector Store")
    auth_token = st.text_input("Auth Token (Vector Store Deletion)")
    if st.button("Delete Vector Store"):
        if auth_token.strip():
            response = requests.delete(f"{BASE_URL}/delete_vector_store/{auth_token}")
            if response.status_code == 200:
                st.success("Vector Store Deleted Successfully!")
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown Error')}")
        else:
            st.error("Please provide an Auth Token.")

# ----------- Tab 3: Update Vector Store -----------
with tabs[2]:
    st.header("Update Vector Store")
    auth_token = st.text_input("Auth Token (Vector Store Update)")
    new_name = st.text_input("New Name for Vector Store")
    if st.button("Update Vector Store"):
        if auth_token.strip() and new_name.strip():
            response = requests.put(
                f"{BASE_URL}/update_vector_store",
                json={"auth_token": auth_token, "new_name": new_name}
            )
            if response.status_code == 200:
                st.success(f"Vector Store Updated to: {new_name}")
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown Error')}")
        else:
            st.error("Please provide both Auth Token and New Name.")

# ----------- Tab 4: Add Vector -----------
with tabs[3]:
    st.header("Add Vector")
    auth_token = st.text_input("Auth Token")
    vector_text = st.text_area("Vector Text")
    subject = st.text_input("Subject")
    metadata = st.text_area("Metadata (JSON format)", value="{}")
    
    if st.button("Add Vector"):
        try:
            metadata_dict = eval(metadata)  # Convert string to dictionary
            response = requests.post(
                f"{BASE_URL}/add_vector",
                json={"auth_token": auth_token, "texts": vector_text, "subject": subject, "metadata": metadata_dict}
            )
            if response.status_code == 200:
                st.success(f"Vector Added Successfully! Vector ID: {response.json()['vector_id']}")
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown Error')}")
        except Exception as e:
            st.error(f"Invalid Metadata Format: {str(e)}")

# ----------- Tab 5: Search Vectors -----------
with tabs[4]:
    st.header("Search Vectors")
    auth_token = st.text_input("Auth Token (Search)")
    query_context = st.text_area("Query Context")
    top_k = st.number_input("Number of Top Results (k)", min_value=1, value=5, step=1)
    
    if st.button("Search Vectors"):
        response = requests.post(
            f"{BASE_URL}/search/{top_k}",
            json={"auth_token": auth_token, "input_query_context": query_context}
        )
        if response.status_code == 200:
            results = response.json()
            for idx, result in enumerate(results):
                st.write(f"**Result {idx+1}:**")
                st.json(result)
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown Error')}")


# ----------- Tab 6: Update Vector -----------
with tabs[5]:
    # ----------- Update Vector Tab -----------
    st.header("Update Vector")

    # Input fields
    auth_token = st.text_input("Auth Token (Update)")
    vector_id = st.text_input("Vector ID")
    new_text = st.text_area("New Text for Vector")

    # Button to compute similarity
    if st.button("Compute Similarity"):
        if not auth_token.strip() or not vector_id.strip() or not new_text.strip():
            st.error("Auth Token, Vector ID, and New Text are required.")
        else:
            # Call the backend to compute similarity
            response = requests.post(
                f"{BASE_URL}/compute_similarity",
                json={"auth_token": auth_token, "vector_id": vector_id, "new_text": new_text}
            )
            if response.status_code == 200:
                # Display similarity score
                similarity = response.json()["similarity"]
                st.session_state["similarity"] = similarity  # Store similarity in session state
                st.session_state["auth_token"] = auth_token
                st.session_state["vector_id"] = vector_id
                st.session_state["new_text"] = new_text

                st.write(f"**Cosine Similarity:** {similarity:.2f}")
            else:
                st.error(f"Error computing similarity: {response.json().get('detail', 'Unknown Error')}")

    # Ensure similarity is computed before proceeding
    if "similarity" in st.session_state:
        st.write(f"**Cosine Similarity:** {st.session_state['similarity']:.2f}")

        # Display options for the user
        action = st.radio(
            "Choose an action for the vector:",
            ["Cancel", "Replace", "Add as New"]
        )

        # # Map user-friendly actions to backend-compatible actions
        # action_map = {
        #     "Cancel": "leave",
        #     "Replace": "edit",
        #     "Add as New": "add"
        # }


        if st.button("Submit Action"):
            if action == "Cancel":
                st.info("No changes will be made to the vector.")
            elif action == "Replace":
                # Make API call to update (Replace) the vector
                update_response = requests.put(
                    f"{BASE_URL}/update_vector",
                    json={
                        "auth_token": st.session_state["auth_token"],
                        "vector_id": st.session_state["vector_id"],
                        "new_text": st.session_state["new_text"],
                        "action": action
                    }
                )
                if update_response.status_code == 200:
                    st.success("Vector Updated Successfully!")
                else:
                    st.error(f"Error: {update_response.json().get('detail', 'Unknown Error')}")
            elif action == "Add as New":
                # Make API call to add a new vector
                add_response = requests.post(
                    f"{BASE_URL}/add_vector",
                    json={
                        "auth_token": st.session_state["auth_token"],
                        "texts": st.session_state["new_text"],
                        "subject": "Newly Added"
                    }
                )
                if add_response.status_code == 200:
                    st.success(f"New Vector Added! Vector ID: {add_response.json()['vector_id']}")
                else:
                    st.error(f"Error: {add_response.json().get('detail', 'Unknown Error')}")




# ----------- Tab 7: Delete Vector -----------
with tabs[6]:
    st.header("Delete Vector")
    auth_token = st.text_input("Auth Token (Delete)")
    vector_id = st.text_input("Vector ID (Delete)")
    
    if st.button("Delete Vector"):
        response = requests.delete(
            f"{BASE_URL}/delete_vector",
            json={"auth_token": auth_token, "vector_id": vector_id}
        )
        if response.status_code == 200:
            st.success("Vector Deleted Successfully!")
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown Error')}")
