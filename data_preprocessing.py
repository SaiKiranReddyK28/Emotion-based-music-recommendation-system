import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from zipfile import ZipFile

# Define paths
data_dir = './data'
raw_data_dir = os.path.join(data_dir, 'raw')
processed_data_dir = os.path.join(data_dir, 'processed')

# Create directories if they don't exist
os.makedirs(raw_data_dir, exist_ok=True)
os.makedirs(processed_data_dir, exist_ok=True)

def unzip_data(file_path, extract_to):
    """Unzip the data file into the specified directory."""
    with ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Unzip raw data (example usage, replace 'your_dataset.zip' with your dataset file)
# If your data is already extracted, comment this out
zip_file_path = os.path.join(raw_data_dir, 'your_dataset.zip')
if os.path.exists(zip_file_path):
    unzip_data(zip_file_path, processed_data_dir)

def preprocess_data():
    """
    Preprocess and augment the data for training.
    This function resizes images, normalizes pixel values, 
    and applies data augmentation for training and validation.
    """
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,  # Split 20% for validation
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Prepare training data generator
    train_data_gen = datagen.flow_from_directory(
        processed_data_dir,    # Directory with the preprocessed data
        target_size=(48, 48),  # Resize images to 48x48
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    # Prepare validation data generator
    val_data_gen = datagen.flow_from_directory(
        processed_data_dir,
        target_size=(48, 48),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_data_gen, val_data_gen

if __name__ == "__main__":
    print("Starting data preprocessing...")
    
    # Preprocess the data
    train_data, val_data = preprocess_data()
    
    print("Data preprocessing complete!")
