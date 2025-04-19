# Deep_Learning-Assignment-2
Implementation of Convolutional Neural Network on inauralist Dataset which has Ten Classes
The Wanbd Report of Project is: https://wandb.ai/ma23c014-indian-institute-of-technology-madras/Convolutional_NN/reports/DA6401-Assignment-2-Convolutional-Neural-Network--VmlldzoxMjM2ODQwNw
Assignment 2: Part A: Training CNN from scratch
In this project i have build a small CNN model consisting of 5 convolution layers. Each convolution layer is followed by an activation and a max-pooling layer.After 5 such conv-activation-maxpool blocks, i have one dense layer followed by the output layer containing 10 neurons(1 for each classes)
The code is flexible such that the number of filters, size of filters, and activation function of the convolution layers and dense layers can be changed.

# Project Structure
* Model Definition: Customizable CNN model with options for activation function, dropout rate, batch normalization, etc.
* Data Augmentation: Implements resizing, horizontal flip, and random rotation using torchvision.transforms.
* Training: Training logic includes tracking metrics like loss and accuracy using WandB.
* Validation: Separate validation loader to track the performance of the model.

## Libraries Used:
1. I have used pytorch to build the CNN model.
2. PIL library was used to handle images.
3. random library was used to select images from test set randomly.
4. os library was used to select images from the desired locations.
5. matplotlib and mpl_toolkits library was used for plotting the (10,3) grid for the predicted images and visualise all the filters in the first layer of our best model for a random image from the test set respectively.
6. 
##  Installations:
I have used pip as the package manager. All the libraries we used above can be installed using the pip command.

## How to use it?
1. Install the requirements: pip install torch, torchvision, sklearn, wandb
2. Use the dataset path in code: Path is- train_dir = "/path/to/dataset/inaturalist_12K/train"
3. Configure and Train: This project uses a sweep configuration to optimize hyperparamters suh as the number of filters, activation functions, dropout etc.

  The sweep config is:-
   sweep_config = {
    "name": "Convolutional_NN", 
    "method": "random",
    "parameters": {
        "num_filters": {"values": [32, 64]},
        "activation_fn": {"values": ["ReLU", "GELU", "SiLU", "Mish"]},
        "dropout_rate": {"values": [0.2, 0.3]},
        "batch_norm": {"values": [True, False]},
        "data_augmentation": {"values": [True, False]},
        "filter_organization": {"values": ["same", "double", "half"]}
    }
}

4. Start Sweep and follow the sweep
5. Evaluate the model: Once training completes, the best model is saved and can be evaluated as follows:-
   ### Load the best model
model = CNNModel(num_classes=<num_classes>, ...)
model.load_state_dict(torch.load("path_to_best_model.pth"))

## Example of Hyperparameter Configuration
Below is an example of how you can pass parameters to the CNN model

model = CNNModel(
    num_classes=10,
    num_filters=64,
    activation_fn=nn.ReLU,
    dropout_rate=0.3,
    batch_norm=True,
    filter_organization="double"
)

# To save best model
wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            }, step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = os.path.join(wandb.run.dir, "best_model.pth")
                torch.save(model.state_dict(), model_path)
                wandb.save(model_path)

# Tools and Libraries
PyTorch: For deep learning model building and training.
Torchvision: For dataset handling and transformations.
Scikit-learn: For train-test splitting.
WandB: For experiment tracking and hyperparameter sweeps.


# PART-B : Fine-tuning a pre-trained model

In this i have instructed to use any one of the pre-trained model(GoogLeNet, InceptionV3, ResNet50, VGG) from torchvision and for the given strategies i have found thw 2nd strategy has more validation ccuracy than other twos. I have used ResNet50 model to train my data.

# Steps
1. Dataset Preparation: The dataset is structured into train and val directories, and image transformations such as resizing, normalization, and tensor conversion are applied using torchvision.transforms.
2. Data Loading:Data is loaded using torch.utils.data.DataLoader to handle batching and shuffling.
3. Model Initialization: The pre-trained ResNet-50 model is loaded, and the fully connected (FC) layer is replaced to match the number of classes in the new dataset.
4. Freezing Layers: Initially, all layers are frozen by setting requires_grad = False.
 * The first k layers (set to 5 in this example) are unfrozen for retraining.
 * Fine-tuning Setup:A new fully connected layer is added to replace the original FC layer.
5. The loss function is set to CrossEntropyLoss. The optimizer used is Adam with a learning rate of 0.0001, applied only to the trainable parameters.
6. Training: The training loop involves: Forward pass, loss computation, and backpropagation for the trainable layers.
    Evaluation of the model on the validation set at the end of each epoch, reporting loss and accuracy.
7. Model Saving: The trained model is saved in the specified path using torch.save.

### How the model is fined-tuned for the inaturalist Dataset

''' # Load pre-trained ResNet-50
model = models.resnet50(weights="ResNet50_Weights.DEFAULT")

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the first `k` layers
k = 5
layers = list(model.children())
for i in range(k):
    for param in layers[i].parameters():
        param.requires_grad = True

# Replace the FC layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Training
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

# Save the model
torch.save(model.state_dict(), "resnet50_finetuned.pth")'''
