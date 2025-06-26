import warnings
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
from torch import nn
warnings.filterwarnings("ignore")


def transform_image(image):
    my_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    return my_transforms(image).unsqueeze(0)


@st.cache_resource  # Кэшируем модель, чтобы не загружать её при каждом обновлении
def load_model():
    model = models.mobilenet_v3_small(pretrained=False)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 66)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5), 
        nn.Linear(1536, 67)
)

    # 3. Now load the checkpoint
    checkpoint = torch.load('model_weights_mobile_net.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()

    return model

# Функция для предсказания
def get_prediction(image, model, labels):
    
    tensor = transform_image(image=image)
    outputs = list(model.forward(tensor).detach().numpy())[0]
    output_dict = dict()

    for i, k in enumerate(outputs):
        output_dict[k] = i
    
    outputs = sorted(outputs, reverse=True)
    result_idx = []
    
    for i in range(5):
        result_idx.append(output_dict[outputs[i]])
    
    result = []

    for elem in result_idx:
        result.append(labels[elem])

    return result


def main():
    st.title("Определение породы кота/кошки")
    st.write("Загрузите изображение котика, и модель предскажет его породу.")
  
    model = load_model()
    labels = ['Abyssinian', 'American Bobtail', 'American Curl', 'American Shorthair', 'American Wirehair', 
              'Applehead Siamese', 'Balinese', 'Bengal', 'Birman', 'Bombay', 'British Shorthair', 'Burmese', 
              'Burmilla', 'Calico', 'Canadian Hairless', 'Chartreux', 'Chausie', 'Chinchilla', 'Cornish Rex', 
              'Cymric', 'Devon Rex', 'Dilute Calico', 'Dilute Tortoiseshell', 'Domestic Long Hair', 'Domestic Medium Hair', 
              'Domestic Short Hair', 'Egyptian Mau', 'Exotic Shorthair', 'Extra-Toes Cat - Hemingway Polydactyl', 'Havana', 
              'Himalayan', 'Japanese Bobtail', 'Javanese', 'Korat', 'LaPerm', 'Maine Coon', 'Manx', 'Munchkin', 'Nebelung', 
              'Norwegian Forest Cat', 'Ocicat', 'Oriental Long Hair', 'Oriental Short Hair', 'Oriental Tabby', 'Persian', 
              'Pixiebob', 'Ragamuffin', 'Ragdoll', 'Russian Blue', 'Scottish Fold', 'Selkirk Rex', 'Siamese', 'Siberian', 
              'Silver', 'Singapura', 'Snowshoe', 'Somali', 'Sphynx - Hairless Cat', 'Tabby', 'Tiger', 'Tonkinese', 'Torbie', 
              'Tortoiseshell', 'Turkish Angora', 'Turkish Van', 'Tuxedo', 'York Chocolate']
    
    # Загрузка изображения
    uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Загруженное изображение", use_container_width=True)
        
        # Предсказание
        if st.button("Классифицировать"):
            st.write("Классификация...")
            prediction = get_prediction(image, model, labels)
            st.success("**Результат (топ 5 возможных пород):**")

            for i in range(len(prediction)):
                st.write(f'{i + 1}. {prediction[i]}')

if __name__ == "__main__":
    main()