from diffusion_prior import *
from custom_pipeline import *
emb_eeg_train = torch.load('D:\医工所\EEG_Image_decode-main\LaBraM for eegthings\checkpoints\labram_eeg_features_sub-01_train.pt')
emb_eeg_test = torch.load('D:\医工所\EEG_Image_decode-main\LaBraM for eegthings\checkpoints\labram_eeg_features_sub-01_test.pt')
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1).to(device)
save_path="D:/医工所/EEG_Image_decode-main/LaBraM for eegthings/checkpoints/diffusion_prior_labrambased.pt"
state_dict = torch.load(save_path, map_location=device)
diffusion_prior.load_state_dict(state_dict)
diffusion_prior.eval()
pipe = Pipe(diffusion_prior, device=device)
from PIL import Image
import os
sub = 'sub-01'
train = False
classes = None
pictures = None
def load_data():
    data_list = []
    label_list = []
    texts = []
    images = []

    if train:
        text_directory = "D:/eeg-things数据集/images_set/training_images"
    else:
        text_directory = "D:/eeg-things数据集/images_set/test_images"

    dirnames = [d for d in os.listdir(text_directory) if os.path.isdir(os.path.join(text_directory, d))]
    dirnames.sort()

    if classes is not None:
        dirnames = [dirnames[i] for i in classes]

    for dir in dirnames:

        try:
            idx = dir.index('_')
            description = dir[idx + 1:]
        except ValueError:
            print(f"Skipped: {dir} due to no '_' found.")
            continue

        new_description = f"{description}"
        texts.append(new_description)

    if train:
        img_directory = "D:/eeg-things数据集/images_set/training_images"
    else:
        img_directory = "D:/eeg-things数据集/images_set/test_images"

    all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
    all_folders.sort()

    if classes is not None and pictures is not None:
        images = []
        for i in range(len(classes)):
            class_idx = classes[i]
            pic_idx = pictures[i]
            if class_idx < len(all_folders):
                folder = all_folders[class_idx]
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()
                if pic_idx < len(all_images):
                    images.append(os.path.join(folder_path, all_images[pic_idx]))
    elif classes is not None and pictures is None:
        images = []
        for i in range(len(classes)):
            class_idx = classes[i]
            if class_idx < len(all_folders):
                folder = all_folders[class_idx]
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()
                images.extend(os.path.join(folder_path, img) for img in all_images)
    elif classes is None:
        images = []
        for folder in all_folders:
            folder_path = os.path.join(img_directory, folder)
            all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.sort()
            images.extend(os.path.join(folder_path, img) for img in all_images)
    else:

        print("Error")
    return texts, images


texts, images = load_data()
# Assuming generator.generate returns a PIL Image
generator = Generator4Embeds(num_inference_steps=4, device=device)
directory = f"generated_imgs/{sub}"
for k in range(200):
    eeg_embeds = emb_eeg_test[k:k+1]#测试集 atms输出的脑电编码1x1024
    h = pipe.generate(c_embeds=eeg_embeds, num_inference_steps=50, guidance_scale=5.0)#由训练好的生成模型初步生成图像编码
    for j in range(4):
        image = generator.generate(h.to(dtype=torch.float16))
        # Construct the save path for each image
        path = f'{directory}/{texts[k]}/{j}.png'
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save the PIL Image
        image.save(path)
        print(f'Image saved to {path}')
