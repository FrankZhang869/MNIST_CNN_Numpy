model = CNN([
    Conv(1, 8, 3, padding=1),
    ReLu(),
    Max_Pool(2),

    Conv(8, 16, kernel_size=3, padding=1),
    ReLu(),
    Max_Pool(2),

    Flatten(),

    Dense(16 * 7 * 7, 128),
    ReLu(),
    Dense(128, 10)
])


for epoch in range(20):
    epoch_loss = 0
    total_correct = 0
    for X_batch, y_batch in train_loader:

        # Forward
        logits = model.forward(X_batch)

        # Loss + gradient
        preds = np.argmax(logits, axis=1)
        total_correct += np.sum(preds == y_batch)
        loss, d_logits = Softmax_Cross_Entropy(logits, y_batch)
        epoch_loss += loss * y_batch.shape[0]
        # Backward
        model.backward(d_logits)

        # Update
        model.step(0.01)
    epoch_loss /= total_sample
    accuracy = total_correct / total_sample
    print(f"Epoch {epoch} | Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.4f}")
