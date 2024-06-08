import matplotlib.pyplot as plt

# Beispielhafte Daten für das Federated Learning Modell
fed_test_accuracies = [0.539, 0.977, 1.0]


plt.figure(figsize=(10, 6))
plt.plot(range(1, len(fed_test_accuracies) + 1), fed_test_accuracies, marker='o', label='Test Accuracy (Federated Learning)')
plt.xlabel('Round/Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy (Federated Learning Model)')
plt.legend()
plt.show()


# Beispielhafte Daten für das Zentrales-Modell
val_accuracies = [0.938, 0.969, 0.984]
test_accuracies = [0.969]

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, marker='o', label='Validation Accuracy (Partitions)')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, marker='x', label='Test Accuracy (Partitions)')
plt.xlabel('Partition')
plt.ylabel('Accuracy')
plt.title('Validation and Test Accuracy (zentrales Model)')
plt.legend()
plt.show()
