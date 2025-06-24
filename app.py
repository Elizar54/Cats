import warnings
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
warnings.filterwarnings("ignore")


def transform_image(image):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225]),
                                        transforms.RandomHorizontalFlip(0.3),
                                        transforms.RandomRotation(180)])
    return my_transforms(image).unsqueeze(0)


# Загрузка предобученной модели
@st.cache_resource  # Кэшируем модель, чтобы не загружать её при каждом обновлении
def load_model():
    model = models.resnet18(pretrained=False)  # or resnet34, resnet50, etc.

    # 2. Modify the final FC layer to match the checkpoint (66 classes)
    num_ftrs = model.fc.in_features  # Get input features of the last layer
    model.fc = torch.nn.Linear(num_ftrs, 66)  # Change output to 66 classes

    # 3. Now load the checkpoint
    checkpoint = torch.load('model_weights_efficient_net.pth', map_location=torch.device('cpu'))
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

# Интерфейс Streamlit
def main():
    st.title("Определение породы кота/кошки с помощью ResNet18")
    st.write("Загрузите изображение котика, и модель предскажет его породу.")
    
    # Загрузка модели и меток
    model = load_model()
    labels = ['abyssinian', 'american_bobtail', 'american_curl', 'american_shorthair', 'american_wirehair', 
           'balinese', 'bengal', 'birman', 'bombay', 'british_shorthair', 'burmese', 'chartreux', 'chausie', 
           'cornish_rex', 'cymric', 'cyprus', 'devon_rex', 'donskoy', 'egyptian_mau', 'european_shorthair', 
           'exotic_shorthair', 'german_rex', 'havana_brown', 'himalayan', 'japanese_bobtail', 'karelian_bobtail', 
           'khao_manee', 'korat', 'korean_bobtail', 'kurilian_bobtail', 'laperm', 'lykoi', 'maine_coon', 'manx', 
           'mekong_bobtail', 'munchkin', 'nebelung', 'norwegian_forest_cat', 'ocicat', 'oregon_rex', 'oriental_shorthair', 
           'persian', 'peterbald', 'pixie_bob', 'ragamuffin', 'ragdoll', 'russian_blue', 'safari', 'savannah', 'scottish_fold', 
           'selkirk_rex', 'serengeti', 'siamese', 'siberian', 'singapura', 'sokoke', 'somali', 'sphynx', 'thai', 'tonkinese', 
           'toyger', 'turkish_angora', 'turkish_van', 'ukrainian_levkoy', 'ural_rex', 'vankedisi']
    
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