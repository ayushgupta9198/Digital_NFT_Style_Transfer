import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import  torchvision.models as models
from torchvision.utils import save_image
import os


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

def load_image(image_name):
    image = Image.open(image_name)
    # print(image.size, type(image)) #print image size before
    image = loader(image).unsqueeze(0)
    # print("after",image.size(), type(image)) #print size after
    return image.to(device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_size = 1080

loader = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[], std=[]),
    ]
)


def generate_style(original_path, style_path,):
    original_image = load_image(original_path)
    style_image = load_image(style_path)

    orig_name = original_path.split("/")[-1].split('.')[0]
    style_name = style_path.split("/")[-1].split('.')[0]
    res_dir = f"/content/drive/MyDrive/Ayush_Gupta/NFT-AI-ART/neural-style-transfer/Images/Output Images/{orig_name}_{style_name}"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    # Initial noisy image
    # generated_image = torch.randn(original_image.shape, device=device, requires_grad=True)
    # Also the generated image can be a clone of the original image [Gives better result]
    generated_image = original_image.clone().requires_grad_(True)

    # model.eval() to freeze the weights
    model = VGG().to(device).eval()

    total_steps = 6000 # Default is 6000
    learning_rate = 1e-3
    alpha = 1       # Content Loss
    beta = 0.01     # Amt of style in the image
    optimizer = optim.Adam([generated_image], lr=learning_rate)

    for step in range(total_steps):
        generated_features = model(generated_image)
        original_features = model(original_image)
        style_features = model(style_image)

        style_loss = 0
        original_loss = 0

        for generated_feature, original_feature, style_feature in zip(
            generated_features, original_features, style_features
        ):
            batch_size, channels, height, width = generated_feature.shape
            original_loss += torch.mean((generated_feature - original_feature) ** 2)

            # Gram Matrix for the generated image
            G = generated_feature.view(channels, height*width).mm(
                generated_feature.view(channels, height*width).t())
            # Gram Matrix for the style image
            A = style_feature.view(channels, height*width).mm(
                style_feature.view(channels, height*width).t())

            style_loss += torch.mean((G-A)**2)

        total_loss = alpha*original_loss + beta*style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step%1000==0:
            print(f'Step: {step}/{total_steps} \t Loss: {total_loss}')
            filename = f"{res_dir}/GeneratedImgStep" + str(step) + ".jpg"
            save_image(generated_image, filename)

    print("Training Complete!")
    filename = f"/content/drive/MyDrive/Ayush_Gupta/NFT-AI-ART/neural-style-transfer/Images/final_images/{orig_name}_{style_name}.jpg"
    save_image(generated_image, filename)

if __name__ == '__main__':
    orig_dir = "/content/drive/MyDrive/Ayush_Gupta/NFT-AI-ART/neural-style-transfer/Images/Input Images/Original Image"
    style_dir = "/content/drive/MyDrive/Ayush_Gupta/NFT-AI-ART/neural-style-transfer/Images/Input Images/Style Image"

    for ct_img in os.listdir(orig_dir):
        if ct_img in ".ipynb_checkpoints":
            print("Taking and Continueing Content Image")
            continue
        for st_img in os.listdir(style_dir):
            if st_img in ".ipynb_checkpoints":
                print("Taking and Continueing Style Image")
                continue
            original_path = os.path.join(orig_dir, ct_img)
            style_path = os.path.join(style_dir, st_img)
            generate_style(original_path, style_path)
