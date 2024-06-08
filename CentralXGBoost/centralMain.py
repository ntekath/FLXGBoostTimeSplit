import xgboost as xgb
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

from dataSet import prepare_dataset


def main():
    # Vorbereitung der Daten
    trainloaders, valloaders, testloaders = prepare_dataset(num_partitions=3, batch_size=128, val_ratio=0.1)

    # Initialisierung des XGBoost-Modells
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        max_depth=3,
        learning_rate=0.03,
        n_estimators=10,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
        tree_method="exact"
    )

    # Initialisierung des Bagging-Modells mit XGBoost-Modell als Basisschätzer
    bagging_model = BaggingClassifier(
        estimator=xgb_model,
        n_estimators=10,
        random_state=42,
        n_jobs=16  # entspricht nthread=16
    )

    # Listen zur Speicherung der Lernkurven-Daten
    train_accuracies = []
    val_accuracies = []

    # Trainieren des Bagging-Modells
    for trainloader, valloader in zip(trainloaders, valloaders):
        X_train, y_train = next(iter(trainloader))
        X_val, y_val = next(iter(valloader))

        bagging_model.fit(X_train.numpy(), y_train.numpy())

        # Auswertung auf dem Trainingsset
        y_pred_train = bagging_model.predict(X_train.numpy())
        train_accuracy = accuracy_score(y_train.numpy(), y_pred_train)
        train_accuracies.append(train_accuracy)


        # Auswertung auf dem Validierungsset
        y_pred_val = bagging_model.predict(X_val.numpy())
        val_accuracy = accuracy_score(y_val.numpy(), y_pred_val)
        val_accuracies.append(val_accuracy)
        val_precision = precision_score(y_val.numpy(), y_pred_val)
        val_f1 = f1_score(y_val.numpy(), y_pred_val)
        val_auc = roc_auc_score(y_val.numpy(), y_pred_val)


        print(f'Validation Accuracy (Bagging): {val_accuracy}')
        print(f'Validation Precision (Bagging): {val_precision}')
        print(f'Validation F1 Score (Bagging): {val_f1}')
        print(f'Validation AUC (Bagging): {val_auc}')

    # Auswertung auf dem Testset für das Bagging-Modell
    test_accuracies = []
    test_precisions = []
    test_f1_scores = []
    test_aucs = []

    for testloader in testloaders:
        X_test, y_test = next(iter(testloader))
        y_pred_test = bagging_model.predict(X_test.numpy())

        test_accuracy = accuracy_score(y_test.numpy(), y_pred_test)
        test_accuracies.append(test_accuracy)

        test_precision = precision_score(y_test.numpy(), y_pred_test)
        test_precisions.append(test_precision)

        test_f1 = f1_score(y_test.numpy(), y_pred_test)
        test_f1_scores.append(test_f1)

        test_auc = roc_auc_score(y_test.numpy(), y_pred_test)
        test_aucs.append(test_auc)

    mean_test_accuracy = sum(test_accuracies) / len(test_accuracies)
    mean_test_precision = sum(test_precisions) / len(test_precisions)
    mean_test_f1 = sum(test_f1_scores) / len(test_f1_scores)
    mean_test_auc = sum(test_aucs) / len(test_aucs)

    print(f'Mean Test Accuracy (Bagging): {mean_test_accuracy}')
    print(f'Mean Test Precision (Bagging): {mean_test_precision}')
    print(f'Mean Test F1 Score (Bagging): {mean_test_f1}')
    print(f'Mean Test AUC (Bagging): {mean_test_auc}')

    # Plot der Lernkurve
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Partition')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve (Zentrales Model)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
