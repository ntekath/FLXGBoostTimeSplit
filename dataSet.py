import numpy as np
import os
import pandas as pd
import torch
import socket
import struct
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit


def count_unique_values(data, column_name):
    unique_values = data[column_name].nunique()
    return unique_values


# Funktion zur Umwandlung von IP-Adressen in numerische Werte
def ip_to_int(ip):
    return struct.unpack("!I", socket.inet_aton(ip))[0]


def get_dataset():
    dataset_path = "C:/Users/noelt/OneDrive/Desktop/Studium/PraxisProjekt/Datensatz2/archive"
    data_file = "CTU-IoT-Malware-Capture-1-1conn.log.labeled.csv"
    data1 = os.path.join(dataset_path, data_file)

    # Lade die Daten
    # Einlesen der Daten
    data = pd.read_csv(data1, sep="|")

    # Test Anzahl der einzigartigen Werte eines Features
    unique_values_count = count_unique_values(data, 'history')
    print(f"Die Spalte 'history' hat {unique_values_count} einzigartige Werte.")

    # Datenbereinigung
    # Entfernen irrelevanter Features
    columns_to_drop = ['tunnel_parents', 'uid', 'missed_bytes', 'local_resp', 'local_orig', 'detailed-label',
                       'history', 'conn_state']
    data.drop(columns=columns_to_drop, inplace=True)

    data[['id.orig_h', 'id.resp_h']] = data[['id.orig_h', 'id.resp_h']].map(ip_to_int)

    # Konvertiere die Zielklasse in numerischen Wert
    target_mapping = {'Benign': 0, 'Malicious': 1}
    data['label'] = data['label'].map(target_mapping)

    # Ersetzen Sie "-" durch NaN (fehlende Werte)
    data.replace("-", np.nan, inplace=True)

    # Fehlende Werte behandeln
    data.fillna(0, inplace=True)

    # Zuerst wählen wir die Spalten aus, die wir one-hot encodieren möchten
    data_to_encode = data[['proto', 'service']]

    # Dann wenden wir den One-Hot-Encoder an
    encoded_data = pd.get_dummies(data_to_encode)

    # Schließlich fügen wir die encodierten Spalten zum ursprünglichen DataFrame hinzu und entfernen die nicht
    # encodierten Spalten
    data = pd.concat([data.drop(columns=['proto', 'service']), encoded_data], axis=1)

    print(data.dtypes)

    # Boolische Werte auf Integer abbilden
    trueFalse_mapping = {True: 1, False: 0}
    columns_to_map = ['proto_icmp', 'proto_tcp', 'proto_udp', 'service_0', 'service_dhcp', 'service_dns',
                      'service_http', 'service_ssh']
    data[columns_to_map] = data[columns_to_map].applymap(lambda x: trueFalse_mapping.get(x, x))

    # Bereinigen der Daten und Konvertieren der Spalten
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                data[col] = data[col].astype(float)
            except ValueError:
                pass  # Wenn die Umwandlung nicht möglich ist, bleiben die Daten unverändert (z. B. Zeichenketten bleiben Zeichenketten)

    print(data.dtypes)

    return data


def time_split_data(X, y, num_splits=5, test_size=0.2):
    """
    Split the data into training, validation, and test sets using TimeSeriesSplit.

    Parameters:
        X (array-like): The feature matrix.
        y (array-like): The target values.
        num_splits (int): The number of splits for TimeSeriesSplit.
        test_size (float): The proportion of the dataset to include in the test split.

    Returns:
        train_val_test_splits (list of tuples): List containing tuples of (train_idx, val_idx, test_idx) for each split.
    """
    tscv = TimeSeriesSplit(n_splits=num_splits)
    train_val_test_splits = []

    for train_val_idx, test_idx in tscv.split(X):
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]

        # Further split train_val into training and validation sets
        num_train_val = len(X_train_val)
        train_size = int(num_train_val * (1 - test_size))

        train_idx, val_idx = next(TimeSeriesSplit(n_splits=2).split(X_train_val[:train_size]))

        train_idx = train_val_idx[train_idx]
        val_idx = train_val_idx[val_idx]

        train_val_test_splits.append((train_idx, val_idx, test_idx))

    return train_val_test_splits


def prepare_dataset(num_partitions: int = 5, batch_size: int = 128, val_ratio: float = 0.1):
    """Prepare dataset by splitting into partitions for federated learning."""

    # Laden des Datensatzes und Vorverarbeitung
    data = get_dataset()

    # Aufteilen der Zievariable
    X = data.drop(columns=['label']).values
    y = data['label'].values

    # Aufteilen der Daten in Trainings-, Validierungs- und Testsets
    train_val_test_splits = time_split_data(X, y, num_splits=num_partitions, test_size=val_ratio)

    # Erstellen von Trainings-, Validierungs- und Test-Datenladern
    trainloaders = []
    valloaders = []
    testloaders = []

    for train_idx, val_idx, test_idx in train_val_test_splits:
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
        trainloaders.append(trainloader)

        val_data = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
        valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
        valloaders.append(valloader)

        test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        testloaders.append(testloader)

    return trainloaders, valloaders, testloaders
