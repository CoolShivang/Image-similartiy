# import streamlit as st
# import cv2
# import numpy as np
# from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.preprocessing.image import img_to_array
# from mtcnn import MTCNN
# from pymilvus import connections, Collection
# from PIL import Image

# # Connect to Milvus
# connections.connect("default", host="localhost", port="19530")
# collection_name = "celeb_image_keras_IVF_FLAT_L2"  # Your collection name

# # Load VGG16 model for feature extraction
# vgg_model = VGG16(weights='imagenet', include_top=False)

# # Function to detect and crop face
# def crop_face(image):
#     detector = MTCNN()
#     faces = detector.detect_faces(image)
#     if faces:
#         x1, y1, width, height = faces[0]['box']
#         x1, y1, width, height = abs(x1), abs(y1), abs(width), abs(height)
#         cropped_face = image[y1:y1 + height, x1:x1 + width]
#         return cv2.resize(cropped_face, (224, 224))  # Resize to 224x224
#     return None

# # Function to get embedding from VGG16 model
# def get_embedding(face):
#     face = img_to_array(face)
#     face = np.expand_dims(face, axis=0)
#     face = preprocess_input(face)
#     embedding = vgg_model.predict(face)
#     return embedding.flatten()  # Flatten to 1D array

# # Function to find similar images in Milvus
# def find_similar_images(embedding, collection_name):
#     collection = Collection(collection_name)
#     # Perform the query to find similar images
#     query_embedding = [embedding.tolist()]  # Convert to list for querying
#     results = collection.query(expr="id >= 0", limit=10, output_fields=["id", "embedding"])  # Change as needed

#     # Calculate distances (L2) and sort results
#     distances = []
#     for result in results:
#         distance = np.linalg.norm(np.array(embedding) - np.array(result['embedding']))
#         distances.append((result['id'], distance))
#     distances.sort(key=lambda x: x[1])  # Sort by distance
#     return distances[:10]  # Return top 10 similar images

# # Streamlit app layout
# st.title("Celebrity Face Recognition")
# uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Convert the uploaded file to an OpenCV image
#     image = Image.open(uploaded_file)
#     image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Detect and crop face
#     face = crop_face(image)
#     if face is not None:
#         st.image(face, caption="Detected Face", use_column_width=True)

#         # Get embedding from VGG16
#         embedding = get_embedding(face)

#         # Find similar images in Milvus
#         similar_images = find_similar_images(embedding, collection_name)

#         # Display similar images
#         st.subheader("Similar Images:")
#         for img_id, distance in similar_images:
#             st.write(f"Image ID: {img_id}, Distance: {distance:.4f}")
#             # Add code to retrieve and display the image if needed
#             # e.g., using the get_image_by_id function from previous code
#     else:
#         st.write("No face detected in the uploaded image.")


# import streamlit as st
# import numpy as np
# import cv2
# from keras.applications import VGG16
# from keras.applications.vgg16 import preprocess_input
# from keras.models import Model
# from pymilvus import Collection, connections
# import os
# from PIL import Image

# # Path to the folder where the images are stored
# image_folder = r'D:\Project\Replaced_Faces'

# # Milvus connection setup
# def connect_to_milvus():
#     connections.connect("default", host="localhost", port="19530")
#     return Collection("celeb_image_keras_IVF_FLAT_L2")

# # Function to preprocess the uploaded image (same preprocessing steps)
# def preprocess_image(image, img_size=224):
#     img = np.array(image)
#     img = cv2.resize(img, (img_size, img_size))  # Resize to 224x224
#     img = img.astype('float32') / 255.0  # Normalize pixel values
#     img = preprocess_input(img)  # Preprocess for VGG16
#     return np.expand_dims(img, axis=0)  # Add batch dimension

# # Feature extraction function using VGG16 model
# def extract_features(image):
#     base_model = VGG16(weights='imagenet', include_top=False)
#     model = Model(inputs=base_model.input, outputs=base_model.output)
#     features = model.predict(image)
#     return features.flatten()  # Flatten the feature map

# # Query Milvus for similar images
# def search_similar_images(collection, features, top_k=3):
#     search_params = {"metric_type": "L2", "params": {"ef": 200}}
#     results = collection.search(
#         data=[features],
#         anns_field="embedding",
#         param=search_params,
#         limit=top_k,
#         output_fields=["id", "label"]
#     )
#     return results

# # Streamlit app setup
# st.title("Celebrity Look-Alike Finder")
# st.write("Upload an image to find similar celebrities from the Milvus database.")

# # Upload an image
# uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_image is not None:
#     # Display the uploaded image
#     st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
#     # Preprocess the uploaded image
#     image = Image.open(uploaded_image)
#     processed_image = preprocess_image(image)

#     # Extract features using VGG16
#     features = extract_features(processed_image)
    
#     # Connect to Milvus and search for similar images
#     collection = connect_to_milvus()
#     search_results = search_similar_images(collection, features)
    
#     st.write(f"Top {len(search_results[0])} similar images:")
    
#     # Display the top 3 similar images
#     for result in search_results[0]:
#         image_id = result.id  # Get the image ID (assuming it is an integer)
#         label = result.entity.get('label')
        
#         # Format the image_id to match the expected filename format (e.g., 000001_face.jpg)
#         image_name = f"{str(image_id).zfill(6)}_face.jpg"  # Pad with zeros to ensure 6 digits, then append '_face'
#         image_path = os.path.join(image_folder, image_name)
        
#         # Display the image and its metadata (ID and label)
#         if os.path.exists(image_path):
#             st.image(image_path, caption=f"Image ID: {image_id}, Celebrity ID: {label}", use_column_width=True)
#         else:
#             st.write(f"Image ID: {image_id}, Celebrity ID: {label} (Image not found)")


# print("All collections created and loaded.")


# import streamlit as st
# import numpy as np
# from pymilvus import Collection, connections
# from PIL import Image
# import os

# # Path to the folder where the images are stored
# image_folder = r'D:\Project\Images'

# # Milvus connection setup
# def connect_to_milvus():
#     connections.connect("default", host="localhost", port="19530")
#     return Collection("celeb_image_keras_IVF_FLAT_L2")  # Update collection name as needed

# # Get the embedding for a specific image ID
# def get_embedding_by_id(image_id, collection_name):
#     collection = Collection(collection_name)
#     collection.load()
    
#     query_expr = f"id == {image_id}"  # Adjust according to your ID field name
#     result = collection.query(expr=query_expr, output_fields=["embedding"])
    
#     if result:
#         return result[0]['embedding']  # Return the embedding of the first result
#     else:
#         raise Exception(f"No embedding found for image ID: {image_id}")

# # Get all embeddings in the collection
# def get_all_embeddings(collection_name, limit=500):
#     collection = Collection(collection_name)
#     collection.load()

#     all_embeddings = []
#     offset = 0

#     while True:
#         batch_embeddings = collection.query(expr="id >= 0", output_fields=["id", "embedding"], offset=offset, limit=limit)
        
#         if not batch_embeddings:
#             break
        
#         all_embeddings.extend(batch_embeddings)
#         offset += limit

#     return all_embeddings

# # Calculate L2 distance between two vectors
# def calculate_l2_distance(vec1, vec2):
#     return np.linalg.norm(np.array(vec1) - np.array(vec2))

# # Find closest images based on L2 distance
# def find_closest_images(image_id, collection_name):
#     target_embedding = get_embedding_by_id(image_id, collection_name)
#     all_embeddings = get_all_embeddings(collection_name)

#     distances = []
#     for entry in all_embeddings:
#         current_id = entry['id']
#         current_embedding = entry['embedding']
        
#         if current_id != image_id:  # Skip the same image
#             distance = calculate_l2_distance(target_embedding, current_embedding)
#             distances.append((current_id, distance))

#     distances.sort(key=lambda x: x[1])  # Sort by distance
#     return distances

# # Streamlit app setup
# st.title("Celebrity Look-Alike Finder")
# st.write("Upload an image to find similar celebrities from the Milvus database.")

# # Upload an image
# uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_image is not None:
#     st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

#     # Assuming you want to find similar images for image ID 0 (you can adjust this)
#     image_id = 0  # Change this to the ID of the uploaded image if needed
#     collection_name = "celeb_image_keras_IVF_FLAT_L2"  # Replace with your collection name
    
#     # Find closest images
#     closest_images = find_closest_images(image_id, collection_name)

#     st.write(f"Top {len(closest_images)} closest images to image ID {image_id}:")

#     # Display the closest images
#     for img_id, dist in closest_images[:10]:  # Display top 10 closest images
#         img_file_name = f"{str(img_id).zfill(6)}.jpg"  # Format the ID
#         img_path = os.path.join('D:\\Project\\Images', img_file_name)  # Your image folder path
        
#         if os.path.exists(img_path):
#             similar_image = Image.open(img_path)
#             st.image(similar_image, caption=f"Image ID: {img_id}, Distance: {dist:.4f}", use_column_width=True)
#         else:
#             st.warning(f"Image file {img_file_name} not found.")













import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from pymilvus import Collection, connections
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

 # Connect to Milvus
connections.connect("default", host="localhost", port="19530")
collection_name = "celeb_image_keras_IVF_FLAT_L2"  # Your collection name

# Load VGG16 model pre-trained on ImageNet, without the classification layers (outside function to avoid reloading)
@st.cache_resource
def load_vgg16_model():
    base_model = VGG16(weights='imagenet', include_top=False)  # Exclude the top classification layer
    model = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)  # Use last convolutional layer
    return model

vgg16_model = load_vgg16_model()  # Load once and reuse

# Preprocessing function (resize, normalize, etc.)
def preprocess_image(image):
    # Convert to RGB in case of grayscale or other format
    image = image.convert('RGB')  
    image_resized = image.resize((224, 224))  # Resize to 224x224 for VGG16
    image_array = keras_image.img_to_array(image_resized)  # Convert to numpy array
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = preprocess_input(image_array)  # Preprocess image for VGG16 (e.g., scale pixels)
    return image_array

# Extract embedding from the uploaded image
def extract_embedding(image):
    # Preprocess the uploaded image
    processed_image = preprocess_image(image)
    
    # Use the VGG16 model to predict the embedding
    embedding = vgg16_model.predict(processed_image)  # Get feature vector (embedding)
    
    # Flatten the embedding to a 1D vector (optional, depending on your needs)
    embedding_flat = embedding.flatten()
    
    return embedding_flat

# Function to get all embeddings from the collection
def get_all_embeddings(collection_name, limit=500):
    collection = Collection(collection_name)
    collection.load()

    all_embeddings = []
    offset = 0

    while True:
        batch_embeddings = collection.query(expr="id >= 0", output_fields=["id", "embedding"], offset=offset, limit=limit)
        
        if not batch_embeddings:
            break
        
        all_embeddings.extend(batch_embeddings)
        offset += limit

    return all_embeddings

# Function to calculate L2 distance
def calculate_l2_distance(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

# Function to find closest images by comparing embeddings
def find_closest_images_by_embedding(target_embedding, collection_name):
    # Get all stored embeddings from Milvus collection
    all_embeddings = get_all_embeddings(collection_name)
    
    distances = []
    for entry in all_embeddings:
        current_id = entry['id']
        current_embedding = entry['embedding']
        
        # Calculate L2 distance between the uploaded image embedding and stored embeddings
        distance = calculate_l2_distance(target_embedding, current_embedding)
        distances.append((current_id, distance))

    # Sort by distance (ascending order)
    distances.sort(key=lambda x: x[1])
    return distances

# Streamlit UI
st.title("Find Similar Celebrity Images")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Convert to OpenCV format
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Extract embedding from the uploaded image
    uploaded_embedding = extract_embedding(image)  # Extract embedding from uploaded image
    st.write(f"Embedding for uploaded image: {uploaded_embedding[:10]}...")  # Display first 10 values for verification

    # Find similar images in the Milvus collection
    collection_name = "celeb_image_keras_IVF_FLAT_L2"  # Your collection name
    closest_images = find_closest_images_by_embedding(uploaded_embedding, collection_name)

    # Display closest images
    st.subheader("Most Similar Images")
    for img_id, dist in closest_images[:10]:  # Display top 10 closest images
        # Construct image file name and path
        img_file_name = f"{str(img_id).zfill(6)}.jpg"  # Format the ID
        img_path = os.path.join('D:\\Project\\Images', img_file_name)  # Your image folder path
        
        if os.path.exists(img_path):
            similar_image = Image.open(img_path)
            st.image(similar_image, caption=f"Image ID: {img_id}, Distance: {dist:.4f}", use_column_width=True)
        else:
            st.warning(f"Image file {img_file_name} not found.")





