import logging
import numpy as np
import pandas as pd
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackContext
from PIL import Image
from io import BytesIO
from datetime import datetime
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
import keras
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from dotenv import load_dotenv,find_dotenv
import os

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Bot token from BotFather
load_dotenv()
TOKEN = os.environ.get("API_TOKEN")

# Load ResNet50 Model
img_width, img_height, _ = 224, 224, 3
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
base_model.trainable = False
model = keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Function to get embedding from an image
def get_embedding(model, img):
    # Reshape
    img = img.resize((img_width, img_height))
    # img to Array
    x = image.img_to_array(img)
    # Expand Dim (1, w, h)
    x = np.expand_dims(x, axis=0)
    # Pre process Input
    x = preprocess_input(x)
    return model.predict(x).reshape(-1)

# Function that get recommendations based on the cosine similarity score
def get_recommender(idx, df, top_n = 3):
    cosine_sim = 1 - pairwise_distances(df, metric='cosine')
    indices = pd.Series(range(len(df)), index=df.index)
    sim_idx    = indices[idx]
    sim_scores = list(enumerate(cosine_sim[sim_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    idx_rec    = [i[0] for i in sim_scores]
    idx_sim    = [i[1] for i in sim_scores]
    
    return indices.iloc[idx_rec].index, idx_sim

# Function to calculate similarity scores
def calculate_similarity_scores(query_embedding, embeddings_dict, top_n=3):
    # Convert embeddings dictionary to list of embeddings and image names
    embeddings_list = []
    image_names = []
    for filename, embedding in embeddings_dict.items():
        embeddings_list.append(embedding)
        image_names.append(filename)
    
    embeddings_array = np.array(embeddings_list)
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity([query_embedding], embeddings_array).flatten()
    
    # Sort similarity scores in descending order
    sim_scores = sorted(list(zip(image_names, cosine_sim)), key=lambda x: x[1], reverse=True)
    
    # Get top N similar images
    sim_scores = sim_scores[:top_n]
    
    return sim_scores

async def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text("Salut! Send me a photo and I will process it to return the most similar scarfs from Carres D'Art IV catalogue.")

async def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Send me a photo and I will process it to return the most similar scarfs from Carres D'Art IV catalogue.")

async def process_image(update: Update, context: CallbackContext) -> None:
    """Process the user-sent image, compute embeddings, and store them."""
    photo_file = await update.message.photo[-1].get_file()
    photo_bytes = await photo_file.download_as_bytearray()

    # Process the image (in this case, just load it)
    output_folder = './logs/'
    os.makedirs(output_folder, exist_ok=True)
    filename = f"{update.message.chat_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    img = Image.open(BytesIO(photo_bytes))
    img.save(output_folder + filename)
    # Get embeddings
    embedding = get_embedding(model, img)

    # Save embeddings to CSV
    df_embs = pd.read_csv('./hermes_db/csv/embeddings_n.csv')

    # Convert the new row to a DataFrame with the same columns as df
    embedding = np.reshape(embedding, [1,2048])
    embedding = pd.DataFrame(embedding, columns=df_embs.columns)

    # Append the new row to the original DataFrame
    df_embs = pd.concat([embedding, df_embs], ignore_index=True)

    df_names = pd.read_csv('./hermes_db/csv/filenames_n.csv').values
    df_names = np.insert(df_names, 0, filename)

    idx, coefs = get_recommender(0, df_embs, top_n = 3)

    img_names = []
    for i in idx:
        img_names.append(df_names[i])

    await update.message.reply_text("🌸💗  Here you are: 💗🌸")
    # Send images back to the user
    for i, img_name in enumerate(img_names):
        # Prepare the image to send
        try:
            img_path = f'./hermes_db/output_rect/{img_name}'
            im_name = img_name.split('.')[0]
            with open(img_path, 'rb') as img_file:
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id, 
                    photo=img_file, 
                    caption= f'{im_name} (~{np.round(100*coefs[i])}%)'
                    )
        except Exception as e:
            logger.error(f"Failed to send photo {img_name}: {e}")


def main() -> None:
    """Start the bot."""
    application = ApplicationBuilder().token(TOKEN).build()

    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, process_image))

    # Start the Bot
    application.run_polling()

if __name__ == '__main__':
    main()
