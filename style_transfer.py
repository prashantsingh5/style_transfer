import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Gram Matrix for Style Loss
def gram_matrix(image):
    N, C, H, W = image.size()
    features = image.view(N * C, H * W)
    gram = torch.mm(features, features.t())
    return gram.div(N * C * H * W)

# Normalization Layer
class NormalizationLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image):
        dev = image.device
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)
        return (image - mean) / std

# Content Loss Layer
class ContentLossLayer(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, pred):
        self.loss = torch.mean((pred - self.target) ** 2)
        return pred

# Style Loss Layer
class StyleLossLayer(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = gram_matrix(target).detach()

    def forward(self, pred):
        pred_gram = gram_matrix(pred)
        self.loss = torch.mean((pred_gram - self.target) ** 2)
        return pred

# Total Variation Loss
class TotalVariationLoss(nn.Module):
    def forward(self, image):
        x_diff = image[:, :, 1:, :] - image[:, :, :-1, :]
        y_diff = image[:, :, :, 1:] - image[:, :, :, :-1]
        return torch.sum(torch.abs(x_diff)) + torch.sum(torch.abs(y_diff))

# Compiler Class
class Compiler:
    def __init__(self, baseModel, contentLayerNames, styleLayerNames, style_weights=None, device='cuda:0'):
        self.baseModel = baseModel.to(device)
        self.contentLayerNames = contentLayerNames
        self.styleLayerNames = styleLayerNames
        self.style_weights = style_weights or {}

    def compile(self, contentImage, styleImage, device='cuda:0'):
        contentImage = contentImage.to(device)
        styleImage = styleImage.to(device)
        contentLayers = []
        styleLayers = []
        model = nn.Sequential(NormalizationLayer())
        i = 0

        for layer in self.baseModel.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu{i}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn{i}'
            else:
                raise RuntimeError(f'Unrecognized layer: {layer._class.name_}')

            model.add_module(name, layer)

            if name in self.contentLayerNames:
                target = model(contentImage).detach()
                content_loss_layer = ContentLossLayer(target)
                model.add_module(f'content_loss_{i}', content_loss_layer)
                contentLayers.append(content_loss_layer)

            if name in self.styleLayerNames:
                target = model(styleImage).detach()
                style_loss_layer = StyleLossLayer(target)
                model.add_module(f'style_loss_{i}', style_loss_layer)
                styleLayers.append(style_loss_layer)

        # Truncate the model after the last relevant layer
        for j in range(len(model) - 1, -1, -1):
            if isinstance(model[j], ContentLossLayer) or isinstance(model[j], StyleLossLayer):
                break
        model = model[:j + 1]

        return model, contentLayers, styleLayers

# Trainer Class
class Trainer:
    def __init__(self, model, contentLayers, styleLayers, device='cuda:0'):
        self.model = model.to(device)
        self.contentLayers = contentLayers
        self.styleLayers = styleLayers
        self.tv_loss = TotalVariationLoss()

    def fit(self, image, epochs=30, alpha=1, beta=1e6, tv_weight=1e-5, device='cuda:0'):
        image = image.to(device)
        optimizer = torch.optim.LBFGS([image.requires_grad_(True)])
        content_losses = []
        style_losses = []
        tv_losses = []
        total_losses = []

        for epoch in range(1, epochs + 1):
            def closure():
                clamped_image = image.clamp(0, 1)
                optimizer.zero_grad()
                self.model(image)

                content_loss = sum([layer.loss for layer in self.contentLayers])
                style_loss = sum([layer.loss for layer in self.styleLayers])
                tv_loss = self.tv_loss(image)

                loss = alpha * content_loss + beta * style_loss + tv_weight * tv_loss
                loss.backward()

                content_losses.append(alpha * content_loss.item())
                style_losses.append(beta * style_loss.item())
                tv_losses.append(tv_weight * tv_loss.item())
                total_losses.append(loss.item())
                return loss

            optimizer.step(closure)

            print(f"Epoch {epoch}/{epochs} --- Total Loss: {total_losses[-1]:.4f}")
            print(f"Content Loss: {content_losses[-1]:.4f} --- Style Loss: {style_losses[-1]:.4f} --- TV Loss: {tv_losses[-1]:.4f}")
            print('---' * 17)

        losses = {'total': total_losses, 'content': content_losses, 'style': style_losses, 'tv': tv_losses}
        return losses, torch.clamp(image, 0, 1)

# Image Loader and Display Functions
def ImageLoader(image_path, size=(512, 512)):
    image = Image.open(image_path).convert("RGB")
    loader = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    image = loader(image).unsqueeze(0)
    return image

def ImageShow(tensor, title, size=(10, 8), save=False):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    if save:
        image.save(f"{title}.jpg")
    plt.figure(figsize=size)
    plt.imshow(image)
    plt.title(title)
    plt.show()

# Load VGG19 Model
vgg19 = models.vgg19(pretrained=True).features.eval()

# Define Layers for Content and Style Loss
contentLayerNames = ['conv4']
styleLayerNames = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

# Load Images
contentImage = ImageLoader(input("Enter the path to the content image: "))
styleImage = ImageLoader(input("Enter the path to the style image: "))

# Compile the Model
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
compiler = Compiler(vgg19, contentLayerNames, styleLayerNames, device=device)
model, contentLayers, styleLayers = compiler.compile(contentImage, styleImage, device)

# Train the Model
trainer = Trainer(model, contentLayers, styleLayers, device=device)
inputImage = contentImage.clone()
losses, outImage = trainer.fit(inputImage, epochs=60, alpha=1, beta=1e6, tv_weight=1e-5, device=device)

# Display Output
ImageShow(outImage, title="Output Image", save=True)