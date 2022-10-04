import torch
import torch.nn.functional as F

def train(training_dataloader, device, model, optimizer, name, epoch):
    for i, (images, labels) in enumerate(training_dataloader):
            images, labels = images.to(device), labels.to(device)
            model.train()
            optimizer.zero_grad()
            output = model(images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()

            update_message = (
                f"\nTrain Epoch: {epoch} | Batch: {i}/{len(training_dataloader)}"
                f"| Loss: {loss.item():.4f}"
            )

            print(update_message)
            with open('./experiments/model_'+name+'.txt', 'a') as f:
                    f.write(update_message)

def validate(validation_dataloader, device, model, epoch):
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in validation_dataloader:
            images, labels = images.to(device), labels.to(device)
            model.eval()
            output = model(images)
            val_loss += F.nll_loss(output, labels, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    val_loss /= len(validation_dataloader.dataset)
    val_acc = correct / len(validation_dataloader.dataset)

    print(f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}")

    return val_loss, val_acc


def train_model(validation_dataloader, training_dataloader, model, optimizer, epochs, device, name):
    with open('./experiments/model_'+name+'.txt', 'w') as f:
                    f.write(f'Experiment file for model: {name} \n Model architecture: {model}')
    best_acc = 0
    best_acc_epoch = 0
    best_model = None

    for epoch in range(epochs):
        train(training_dataloader, device, model, optimizer, name, epoch)
        val_loss, val_acc = validate(validation_dataloader, device, model, epoch)

        with open('./experiments/model_'+name+'.txt', 'a') as f:
            f.write(f"\n Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_acc_epoch = epoch
            best_model = model

    with open('./experiments/model_'+name+'.txt', 'a') as f:
                    f.write(f"\n\n Best validation accuracy: {best_acc} occured at epoch: {best_acc_epoch}") 
    # save model
    torch.save(best_model.state_dict(), './experiments/best_model_'+name) 

    return model