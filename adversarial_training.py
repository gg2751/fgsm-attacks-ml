import torch 
import torch.nn as nn 
import torch.optim as optim
from simplednn import SimpleNN, load_data, evaluate
from fgsm_attacks import fgsm_attack, targeted_fgsm_attack

# Model architecture
input_size = 28 * 28  # MNIST images are 28x28 pixels
hidden_size = 128
output_size = 10
batch_size = 64

train_loader, test_loader = load_data(batch_size)

def adversarial_train(type, train_loader):
    model_retrain = SimpleNN(input_size, hidden_size, output_size)
    criterion_retrain = nn.CrossEntropyLoss()
    optimizer_retrain = optim.Adam(model_retrain.parameters(), lr=0.01)

    epochs = 10
    epsilon_retraining = 0.5


    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs = inputs.view(-1, input_size)
            # Perform untargeted FGSM attack during training
            if type=='untargeted':
                adv_inputs = fgsm_attack(model_retrain, inputs, labels, epsilon_retraining)
            elif type=='targeted':
                adv_inputs = targeted_fgsm_attack(model_retrain, inputs, labels, epsilon_retraining)

            inputs_combined = torch.cat((inputs, adv_inputs), 0)
            labels_combined = torch.cat((labels, labels), 0)

            outputs = model_retrain(inputs_combined)
            loss = criterion_retrain(outputs, labels_combined)

            optimizer_retrain.zero_grad()
            loss.backward()
            optimizer_retrain.step()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    return model_retrain

def main():
    print("\nTraining untargeted FGSM attacks model...")
    fgsm_model = adversarial_train('untargeted', train_loader)
    correct, total, accuracy = evaluate(fgsm_model, test_loader)
    torch.save(fgsm_model.state_dict(), './models/untargeted-fgsm-model.pt')

    print("\nTraining targeted FGSM attacks model...")
    targeted_fgsm_model = adversarial_train('targeted', train_loader)
    correct, total, accuracy = evaluate(targeted_fgsm_model, test_loader)
    torch.save(targeted_fgsm_model.state_dict(), './models/targeted-fgsm-model.pt')

if __name__ == "__main__":
    main()

"""
OUTPUT

Training untargeted FGSM attacks model...
Epoch [1/10], Loss: 0.2964
Epoch [2/10], Loss: 0.1271
Epoch [3/10], Loss: 0.1370
Epoch [4/10], Loss: 0.2530
Epoch [5/10], Loss: 0.3106
Epoch [6/10], Loss: 0.2279
Epoch [7/10], Loss: 0.2964
Epoch [8/10], Loss: 0.1177
Epoch [9/10], Loss: 0.1166
Epoch [10/10], Loss: 0.1301
Test Accuracy: 94.59%
Test Correct/Total: 9459/10000

Training targeted FGSM attacks model...
Epoch [1/10], Loss: 0.0180
Epoch [2/10], Loss: 0.2318
Epoch [3/10], Loss: 0.1205
Epoch [4/10], Loss: 0.1318
Epoch [5/10], Loss: 0.1073
Epoch [6/10], Loss: 0.1572
Epoch [7/10], Loss: 0.0251
Epoch [8/10], Loss: 0.0585
Epoch [9/10], Loss: 0.1486
Epoch [10/10], Loss: 0.1966
Test Accuracy: 93.84%
Test Correct/Total: 9384/10000
"""