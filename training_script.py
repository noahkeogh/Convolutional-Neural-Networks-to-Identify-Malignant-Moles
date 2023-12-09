import torch 
from torch import nn 
from tqdm.auto import tqdm

# Create a train step
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device):

  # Put the model in train mode
  model.train()

  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0

  # Loop through data loader data batched
  for batch, (X, y) in enumerate(dataloader):
    # Send data to the target device
    X, y = X.to(device), y.to(device)

    # 1) Forward Pass
    y_pred = model(X).squeeze()
    # y_pred = torch.round(torch.sigmoid(y_logits))

    # 2) Calculate the loss
    loss = loss_fn(y_pred, y.float())
    train_loss += loss.item()

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss Backward
    loss.backward()

    # 5) Optimizer step
    optimizer.step()

    # Calculate accuracy metric
    y_pred_class = torch.round(torch.sigmoid(y_pred))
    train_acc += (y_pred_class==y).sum().item()/len(y_pred)

  # Adjust metrics to get average loss and accuracy per batch
  train_loss /= len(dataloader)
  train_acc /= len(dataloader)
  return train_loss, train_acc

# Create a test step
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device):
  # Put the model in eval mode
  model.eval()

  # Setup test loss and test accuracy values
  test_loss, test_acc = 0, 0

  # Turn on inference mode
  with torch.inference_mode():
    # Loop through DataLoader batches
    for batch, (X, y) in enumerate(dataloader):
      # Send data to target device
      X, y = X.to(device), y.to(device)

      # 1) Forward Pass
      test_pred_logits = model(X).squeeze()

      # 2) Calculate the loss
      loss = loss_fn(test_pred_logits, y.float().view_as(test_pred_logits))
      test_loss += loss.item()

      # 3) Calculate the accuracy
      test_pred_labels = torch.round(torch.sigmoid(test_pred_logits))
      test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc

# Train function
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int, 
          device: str):

  # Create Results dictionary
  results = {"train_loss": [],
             "train_acc": [],
             "val_loss": [],
             "val_acc": []
             }

  # Loop through the training and testing steps
  for epoch in tqdm(range(epochs)):

    # Calculate the training loss and accuracy
    train_loss, train_acc = train_step(model=model,
                                       dataloader=train_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       device=device)

    # Calculate the testing loss and accuracy
    val_loss, val_acc = test_step(model=model,
                                  dataloader=val_dataloader,
                                  loss_fn=loss_fn,
                                  device=device)

    # Print the Status of the training
    print(f"Epoch: {epoch+1} || Train Loss: {train_loss:.5f} || Train Accuracy: {train_acc:.5f} || " +
          f"Validation Loss: {val_loss:.5f} || Validation Accuracy: {val_acc:.5f}")

    # Save the results to the dictionary
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["val_loss"].append(val_loss)
    results["val_acc"].append(val_acc)

  # return the results dictionary
  return results, model.state_dict()

# Make a test script that will be used for the final assesment of the 
# the model on the test data 
def test(model: torch.nn.Module,
         dataloader: torch.utils.data.DataLoader,
         loss_fn: torch.nn.Module,
         device):
  # Put the model in eval mode
  model.eval()

  # Setup test loss and test accuracy values
  test_loss, test_acc = 0, 0
  # Setup values for keeping tack of
  # True Pos, True Neg, False Neg, False Pos
  TP, TN = 0, 0
  FN, FP = 0, 0
  preds = [] 
  labels = [] 

  # Turn on inference mode
  with torch.inference_mode():
    # Loop through DataLoader batches
    for batch, (X, y) in enumerate(dataloader):
      # Send data to target device
      X, y = X.to(device), y.to(device)

      # 1) Forward Pass
      test_pred_logits = model(X).squeeze()

      # 2) Calculate the loss
      loss = loss_fn(test_pred_logits, y.float().view_as(test_pred_logits))
      test_loss += loss.item()

      # 3) Calculate the accuracy
      test_pred_labels = torch.round(torch.sigmoid(test_pred_logits))
      test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

      # Save predictions and labels 
      preds += (torch.sigmoid(test_pred_logits)).tolist()
      labels += y.tolist() 

      # 4) Calculate TP, TN, FP, FN
      TP += torch.logical_and(test_pred_labels == y, y==1).sum().item()
      TN += torch.logical_and(test_pred_labels == y, y==0).sum().item()
      FN += torch.logical_and(test_pred_labels != y, y==1).sum().item()
      FP += torch.logical_and(test_pred_labels != y, y==0).sum().item()

    # Adjust metrics to get average loss and accuracy per batch
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc, TP, TN, FN, FP, preds, labels

