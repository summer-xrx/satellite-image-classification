import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from dataloader import train_loader, test_loader,test_dataset
import torch.nn.functional as F
from concrete.ml.torch.compile import compile_torch_model
import numpy as np
import time
from tqdm import tqdm


device = torch.device('cpu')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=7, stride=2)
        self.conv2 = nn.Conv2d(8, 12, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(12, 8, kernel_size=3, stride=1)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(72,10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = x.view(-1, 72)
        x = self.fc1(x)
        return x

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the model on the 4000 test images: {100 * correct / total:.2f}%')
    return 100*correct/total


#def train(model, train_loader, criterion, optimizer, scheduler, device, epochs=5):
def train(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train()
    max_accurrancy_rate = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 25 == 24:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0
        accurrancy_rate = test(model, test_loader, device)
        torch.save(model.state_dict(),f"./models_new/model_epoch{epoch}_{accurrancy_rate}.pth")
        #test(model, test_loader, device)
    #scheduler.step()

def test_with_concrete(quantized_module, test_loader, use_sim):
    all_y_pred = np.zeros((len(test_loader.dataset)), dtype=np.int64)
    all_targets = np.zeros((len(test_loader.dataset)), dtype=np.int64)
    idx = 0
    for data, target in tqdm(test_loader):
        data = data.numpy()
        target = target.numpy()
        fhe_mode = "simulate" if use_sim else "execute"
        y_pred = quantized_module.forward(data, fhe=fhe_mode)
        endidx = idx + target.shape[0]
        all_targets[idx:endidx] = target
        y_pred = np.argmax(y_pred, axis=1)
        all_y_pred[idx:endidx] = y_pred
        idx += target.shape[0]
    n_correct = np.sum(all_targets == all_y_pred)
    return n_correct / len(test_loader.dataset)


model = CNN().to(device)

model.load_state_dict(torch.load('models_new64/model_epoch84_70.5.pth'))
test(model, test_loader, device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0005)
#optim_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.8)


#train(model, train_loader, criterion, optimizer, optim_scheduler, device, epochs=1)
# train(model, train_loader, criterion, optimizer, device, epochs=100)
#exit()

#model = CNN()
#model.load_state_dict(torch.load('models/model_epoch4.pht'))
# model=torch.load('models\model_epoch29.pht')

#model = CNN().to(device)
#model.load_state_dict(torch.load('models/model.pth'),strict=False)

#test(model, test_loader, device)

#import torch.onnx
#dummy_input = torch.randn(32, 1, 128, 128)
#torch.onnx.export(model, dummy_input, "model.onnx", do_constant_folding=True)

train_features = []
train_labels = []

for inputs, labels in train_loader:
    train_features.append(inputs)
    train_labels.append(labels)

train_features = torch.cat(train_features) #tensor
train_labels = torch.cat(train_labels)

x_train = train_features.to(device)#.numpy()
y_train = train_labels.to(device)#.numpy()

n_bits = 6

test_features = []
test_labels = []

for inputs, labels in test_loader:
    test_features.append(inputs)
    test_labels.append(labels)

test_features = torch.cat(test_features) #tensor
test_labels = torch.cat(test_labels)

x_test = test_features.to(device)#.numpy()
y_test = test_labels.to(device)#.numpy()

print("===================Start Compile========================")
q_module = compile_torch_model(model, x_train[:,:], n_bits=n_bits,rounding_threshold_bits={"n_bits": n_bits+1, "method": "approximate"})
# # q_module = compile_torch_model(model, x_train, n_bits=6,rounding_threshold_bits={"n_bits": 6, "method": "approximate"})


print(q_module.fhe_circuit.statistics)

start_time = time.time()
accs = test_with_concrete(
    q_module,
    test_loader,
    use_sim=True,
)
sim_time = time.time() - start_time

print(f"Simulated FHE execution for {n_bits} bit network accuracy: {(100*accs):.2f}%")
 # Generate keys first
t = time.time()
q_module.fhe_circuit.keygen()
print(f"Keygen time: {time.time()-t:.2f}s")
# Run inference in FHE on a single encrypted example
mini_test_dataset = TensorDataset(torch.Tensor(x_test[:1, :]), torch.Tensor(y_test[:1]))
mini_test_dataloader = DataLoader(mini_test_dataset)

t = time.time()
accuracy_test = test_with_concrete(
    q_module,
    mini_test_dataloader,
    use_sim=False,
)
elapsed_time = time.time() - t
time_per_inference = elapsed_time / len(mini_test_dataset)
accuracy_percentage = 100 * accuracy_test

print(
    f"Time per inference in FHE: {time_per_inference:.2f} "
    f"with {accuracy_percentage:.2f}% accuracy")

