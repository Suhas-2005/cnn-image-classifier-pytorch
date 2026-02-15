import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import CIFAR10CNN
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import os
import gc

def check_cifar10_exists(root):
    """Check if CIFAR-10 dataset is already downloaded and extracted."""
    cifar_path = os.path.join(root, 'cifar-10-batches-py')
    if os.path.exists(cifar_path):
        required_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
        all_files_exist = all(os.path.exists(os.path.join(cifar_path, fname)) for fname in required_files)
        if all_files_exist:
            print("CIFAR-10 dataset already exists. Skipping download.")
            return True
    return False

def download_with_retry(dataset_class, root, train, transform, max_retries=5):
    # First check if dataset already exists
    if check_cifar10_exists(root):
        return dataset_class(root=root, train=train, download=False, transform=transform)
    
    # If not, try downloading
    for attempt in range(max_retries):
        try:
            return dataset_class(root=root, train=train, download=True, transform=transform)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Download attempt {attempt + 1} failed. Retrying in 5 seconds...")
                print(f"Error: {str(e)}")
                time.sleep(5)
            else:
                raise e

def save_checkpoint(epoch, model, optimizer, best_acc, train_losses, test_losses):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'train_losses': train_losses,
        'test_losses': test_losses
    }
    torch.save(checkpoint, 'checkpoint.pth')

def load_checkpoint():
    if os.path.exists('checkpoint.pth'):
        print("Loading from checkpoint...")
        checkpoint = torch.load('checkpoint.pth')
        return checkpoint
    return None

def train_model(epochs=30, batch_size=64, learning_rate=0.001):  # Reduced batch size
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Data transformations
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        print("Downloading and preparing the CIFAR-10 dataset...")
        
        # Create data directory if it doesn't exist
        os.makedirs('./data', exist_ok=True)
        
        # Load CIFAR-10 dataset with retry logic
        trainset = download_with_retry(torchvision.datasets.CIFAR10, './data', 
                                     train=True, transform=transform_train)
        testset = download_with_retry(torchvision.datasets.CIFAR10, './data', 
                                    train=False, transform=transform_test)

        # Create data loaders with reduced number of workers
        trainloader = DataLoader(trainset, batch_size=batch_size,
                               shuffle=True, num_workers=0)
        testloader = DataLoader(testset, batch_size=batch_size,
                              shuffle=False, num_workers=0)

        print("Dataset preparation completed. Starting training...")

        # Initialize model, loss function, and optimizer
        model = CIFAR10CNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

        # Load checkpoint if exists
        start_epoch = 0
        train_losses = []
        test_losses = []
        best_acc = 0.0
        
        checkpoint = load_checkpoint()
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            train_losses = checkpoint['train_losses']
            test_losses = checkpoint['test_losses']
            print(f"Resuming from epoch {start_epoch+1} with best accuracy: {best_acc:.2f}%")

        # Training loop
        try:
            for epoch in range(start_epoch, epochs):
                model.train()
                running_loss = 0.0
                progress_bar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}')
                
                for i, data in enumerate(progress_bar):
                    try:
                        inputs, labels = data[0].to(device), data[1].to(device)
                        
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        
                        running_loss += loss.item()
                        progress_bar.set_postfix({'loss': running_loss/(i+1)})
                        
                        # Clear some memory
                        del outputs, loss
                        if i % 100 == 0:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                    except Exception as e:
                        print(f"Error in training batch: {str(e)}")
                        continue
                
                train_losses.append(running_loss/len(trainloader))
                
                # Validation
                model.eval()
                test_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for data in testloader:
                        try:
                            images, labels = data[0].to(device), data[1].to(device)
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                            test_loss += loss.item()
                            
                            _, predicted = outputs.max(1)
                            total += labels.size(0)
                            correct += predicted.eq(labels).sum().item()
                            
                            # Clear some memory
                            del outputs, loss
                        except Exception as e:
                            print(f"Error in validation batch: {str(e)}")
                            continue
                
                test_loss = test_loss/len(testloader)
                test_losses.append(test_loss)
                accuracy = 100. * correct / total
                
                print(f'Epoch {epoch+1}: Train Loss: {train_losses[-1]:.3f}, '
                      f'Test Loss: {test_loss:.3f}, Accuracy: {accuracy:.2f}%')
                
                # Save best model
                if accuracy > best_acc:
                    print(f'Saving best model with accuracy: {accuracy:.2f}%')
                    torch.save(model.state_dict(), 'cifar10_best.pth')
                    best_acc = accuracy
                
                # Save checkpoint
                save_checkpoint(epoch + 1, model, optimizer, best_acc, train_losses, test_losses)
                
                scheduler.step(test_loss)
                
                # Clear memory at the end of each epoch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Plot training curves
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(test_losses, label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('training_curves.png')
            plt.close()
            
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving checkpoint...")
            save_checkpoint(epoch + 1, model, optimizer, best_acc, train_losses, test_losses)
            print("Checkpoint saved. You can resume training later.")
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        if 'epoch' in locals():
            print("Attempting to save checkpoint before exit...")
            save_checkpoint(epoch + 1, model, optimizer, best_acc, train_losses, test_losses)

if __name__ == "__main__":
    train_model() 