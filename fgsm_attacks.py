import torch
from simplednn import SimpleNN, load_data, evaluate

# Model architecture
input_size = 28 * 28  # MNIST images are 28x28 pixels
hidden_size = 128
output_size = 10
batch_size = 64

model = SimpleNN(input_size, hidden_size, output_size)

# Load the saved model 
model_path = "./models/base-model.pt"
model.load_state_dict(torch.load(model_path))

# Inference mode
model.eval()

def fgsm_attack(model, images, labels, epsilon):
    """
    FGSM Attack: 
        Collect gradient of the loss with respect to the input
        to create perturbed image using the gradient and epsilon.
    """
    
    images.requires_grad = True

    outputs = model(images)
    loss = torch.nn.functional.cross_entropy(outputs, labels)

    model.zero_grad()
    loss.backward()
    
    data_grad = images.grad.data
    perturbed_images = images + epsilon * torch.sign(data_grad)
    perturbed_images = torch.clamp(perturbed_images, 0, 1)  # Ensure values are within valid range

    return perturbed_images

def targeted_fgsm_attack(model, images, labels, epsilon):
    images.requires_grad = True

    outputs = model(images)
    loss = torch.nn.functional.cross_entropy(outputs, labels)

    model.zero_grad()
    loss.backward()

    data_grad = images.grad.data

    perturbed_images = images - epsilon * torch.sign(data_grad)
    perturbed_images = torch.clamp(perturbed_images, 0, 1)  # Ensure values are within valid range

    return perturbed_images

def fgsm_eval(attack_type, model, test_loader, test_correct, test_total):
    epsilon_values = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    for epsilon in epsilon_values:
        adversarial_correct = 0
        for inputs, labels in test_loader:
            inputs = inputs.view(-1, input_size)
            if attack_type == "untargeted":
                adv_inputs = fgsm_attack(model, inputs, labels, epsilon)
            elif attack_type == "targeted":
                # Generate random target labels different from original labels
                target_labels = (labels + 1) % output_size
                adv_inputs = targeted_fgsm_attack(model, inputs, target_labels, epsilon)
            outputs = model(adv_inputs)
            _, predicted = torch.max(outputs.data, 1)

            adversarial_correct += (predicted == labels).sum().item()

        adversarial_accuracy = adversarial_correct / test_total
        success_rate = (test_correct - adversarial_correct) / test_correct

        print(f'Epsilon: {epsilon}, Test Correct: {test_correct}, Adversarial Correct: {adversarial_correct}, Adversarial Accuracy: {adversarial_accuracy*100:.2f}, Success Rate: {success_rate * 100:.2f}%')

def main():
    _ , test_loader = load_data(batch_size)
    test_correct, test_total, _ = evaluate(model, test_loader)
    print("\nUntargeted FGSM attacks:")
    fgsm_eval("untargeted", model, test_loader, test_correct, test_total)
    print("\nTargeted FGSM attacks:")
    fgsm_eval("targeted", model, test_loader, test_correct, test_total)

if __name__ == "__main__":
    main()

"""
OUTPUT

Test Accuracy: 96.48%
Test Correct/Total: 9648/10000

Untargeted FGSM attacks:
Epsilon: 0, Test Correct: 9648, Adversarial Correct: 8954, Adversarial Accuracy: 89.54, Success Rate: 7.19%
Epsilon: 0.01, Test Correct: 9648, Adversarial Correct: 8884, Adversarial Accuracy: 88.84, Success Rate: 7.92%
Epsilon: 0.02, Test Correct: 9648, Adversarial Correct: 8829, Adversarial Accuracy: 88.29, Success Rate: 8.49%
Epsilon: 0.03, Test Correct: 9648, Adversarial Correct: 8778, Adversarial Accuracy: 87.78, Success Rate: 9.02%
Epsilon: 0.04, Test Correct: 9648, Adversarial Correct: 8712, Adversarial Accuracy: 87.12, Success Rate: 9.70%
Epsilon: 0.05, Test Correct: 9648, Adversarial Correct: 8646, Adversarial Accuracy: 86.46, Success Rate: 10.39%
Epsilon: 0.1, Test Correct: 9648, Adversarial Correct: 8255, Adversarial Accuracy: 82.55, Success Rate: 14.44%
Epsilon: 0.2, Test Correct: 9648, Adversarial Correct: 7227, Adversarial Accuracy: 72.27, Success Rate: 25.09%
Epsilon: 0.3, Test Correct: 9648, Adversarial Correct: 5989, Adversarial Accuracy: 59.89, Success Rate: 37.92%
Epsilon: 0.4, Test Correct: 9648, Adversarial Correct: 4765, Adversarial Accuracy: 47.65, Success Rate: 50.61%
Epsilon: 0.5, Test Correct: 9648, Adversarial Correct: 3709, Adversarial Accuracy: 37.09, Success Rate: 61.56%

Targeted FGSM attacks:
Epsilon: 0, Test Correct: 9648, Adversarial Correct: 8954, Adversarial Accuracy: 89.54, Success Rate: 7.19%
Epsilon: 0.01, Test Correct: 9648, Adversarial Correct: 8926, Adversarial Accuracy: 89.26, Success Rate: 7.48%
Epsilon: 0.02, Test Correct: 9648, Adversarial Correct: 8899, Adversarial Accuracy: 88.99, Success Rate: 7.76%
Epsilon: 0.03, Test Correct: 9648, Adversarial Correct: 8876, Adversarial Accuracy: 88.76, Success Rate: 8.00%
Epsilon: 0.04, Test Correct: 9648, Adversarial Correct: 8846, Adversarial Accuracy: 88.46, Success Rate: 8.31%
Epsilon: 0.05, Test Correct: 9648, Adversarial Correct: 8808, Adversarial Accuracy: 88.08, Success Rate: 8.71%
Epsilon: 0.1, Test Correct: 9648, Adversarial Correct: 8602, Adversarial Accuracy: 86.02, Success Rate: 10.84%
Epsilon: 0.2, Test Correct: 9648, Adversarial Correct: 7962, Adversarial Accuracy: 79.62, Success Rate: 17.48%
Epsilon: 0.3, Test Correct: 9648, Adversarial Correct: 6998, Adversarial Accuracy: 69.98, Success Rate: 27.47%
Epsilon: 0.4, Test Correct: 9648, Adversarial Correct: 5782, Adversarial Accuracy: 57.82, Success Rate: 40.07%
Epsilon: 0.5, Test Correct: 9648, Adversarial Correct: 4436, Adversarial Accuracy: 44.36, Success Rate: 54.02%
"""