import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score, auc
import sys 
import torch


try:
  from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
  from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
  from pytorch_grad_cam.utils.image import show_cam_on_image
except:
  import subprocess
  #pip install grad-cam
  subprocess.call([sys.executable, '-m', 'pip', 'install', 'grad-cam'])
  
  from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
  from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
  from pytorch_grad_cam.utils.image import show_cam_on_image


def visualize_example_img(image_path): 
  """
  Given a path to the directory containing the Test and Train Image 
  folders will return an example image from the dataset. 
  Input: 
    image_path: Path object to folder containing the train and test 
                image directories 
  Returns: 
    img: an Image object which is an example image from the dataset 
  """
  # Get all the image paths
  image_path_list = list(image_path.glob("*/*/*.jpg"))
  # Get a random image
  random_image_path = random.choice(image_path_list)
  # Get image class from path name
  image_class = random_image_path.parent.stem
  # Open the image
  img = Image.open(random_image_path)
  # Print Information about the image and visualize it 
  print(f"Random image path: {random_image_path}")
  print(f"Mole Class: {image_class}")
  print(f"Image Height: {img.height}")
  print(f"Image Width: {img.width}")
  return img
  
  
def plot_roc_curve(y_test, y_probs):
  """
  Plots an ROC Curve.
  Inputs:
    y_test: the ground truth labels.
    y_probs: predicted label probabilities.
  Outputs:
    An ROC ruve.
  """
  # Calculate the Area Under Curve (AUC)
  fpr, tpr, thresholds = roc_curve(y_test, y_probs)
  roc_auc = auc(fpr, tpr)
  print("AUC:", roc_auc)

  # Plot the ROC curve
  fig, axs = plt.subplots()
  axs.plot(fpr, tpr,
          color="orange",
          lw=2,
          label=f"Final Model (Area Under Curve = {np.round(roc_auc, 4)})")

  axs.plot([0, 1], [0, 1],
          color="darkblue",
          lw=2,
          linestyle='--',
          label="Random Classifier")

  axs.plot([0, 1], [1, 1],
          color="green",
          lw=2,
          linestyle='--',
          label="Perfect Classifier")
  axs.plot([0, 0], [0, 1],
          color="green",
          lw=2,
          linestyle='--')

  axs.set_xlim([-0.01, 1.0])
  axs.set_ylim([0.0, 1.05])
  axs.set_xlabel('False Positive Rate')
  axs.set_ylabel('True Positive Rate')
  axs.set_title('Receiver Operating Characteristic (ROC) Curve')
  plt.legend(loc="lower right")
  plt.show()


def get_performance_metrics(test_fn, model, test_dataloader, loss_fn, device):
  """
  Prints evaluation metrics for the given model on the test dataset.
  Inputs:
    model: the trained model to evaluate
    test_dataloader: the dataloader with the test data
    loss_fn: loss function to use
    device: which device to use
  """
  # Calculate how well the model performed
  loss, acc, TP, TN, FN, FP, y_probs, y_test = test_fn(model,
                                                       test_dataloader,
                                                       loss_fn,
                                                       device)

  print(f"Model had a Loss of: {loss}")
  print(f"Model had an accuracy of: {acc}")
  print(f"True Positive (TP): {TP}")
  print(f"True Negative (TN): {TN}")
  print(f"False Negative (FN): {FN}")
  print(f"False Positive (FP): {FP}")
  print()

  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  F1 = (2*precision*recall) / (precision + recall)
  accuracy = (TP + TN) / (TP + FN + TN + FP)
  specificity = TN / (TN + FP)

  print(f"Precision: {precision}")
  print(f"Recall: {recall}")
  print(f"F1 Score: {F1}")
  print(f"Accuracy: {accuracy}")
  print(f"Specificity: {specificity}")

  plot_roc_curve(y_test, y_probs)
  

def visualize_image_with_gradcam(dataset, index, model, target_layer, device):
  """
  Returns a GradCAM visualization of an example image from the given dataset,
  with the given model and target layer.

  Inputs:
    dataset: dataset the desired image is in
    index: index of the image in the dataset
    model: model to use for predict the class of the image
    target_layer: model layer to visualize with GradCam
    device: which device to use

  Returns:
    The logits with the predicted and ground truth class and
    the GradCam visualization.
  """
  # Select a particular image to look at
  img = dataset[index][0]
  label = dataset[index][1]

  # package target_layer into a list
  target_layer = [target_layer]
  input_tensor = img.unsqueeze(0)

  # Construct the CAM object then use on many images
  cam = GradCAM(model=model, target_layers=target_layer, use_cuda=True)
  targets = None
  grayscale_cam = cam(input_tensor=input_tensor,
                      targets=targets,
                      aug_smooth=True
                    )

  # Grayscale overlay
  grayscale_cam = grayscale_cam[0, :]

  # Preparing Original Input Image
  rgb_img = img.squeeze(dim=0)

  # Do so by dividing by making
  # min 0 and max 1 for each channel
  chanel1 = rgb_img[0,:,:]
  chanel2 = rgb_img[1,:,:]
  chanel3 = rgb_img[2,:,:]

  rgb_img[0,:,:] = (chanel1 - torch.min(chanel1)) / (torch.max(chanel1) - torch.min(chanel1))
  rgb_img[1,:,:] = (chanel2 - torch.min(chanel2)) / (torch.max(chanel2) - torch.min(chanel2))
  rgb_img[2,:,:] = (chanel3 - torch.min(chanel3)) / (torch.max(chanel3) - torch.min(chanel3))

  # Change Channel Arrangment and make numpy for imshow
  rgb_img = img.squeeze(dim=0).permute(1,2,0).numpy().astype(np.float32)
  # Generate the Visualization
  visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

  # Get the class index dictionary
  class_idx = dataset.class_to_idx
  idx_class = {v: k for k, v in class_idx.items()}

  # Perform a forward pass on a single image
  model.eval()
  with torch.inference_mode():
    pred = model(input_tensor.to(device))

  pred_label = int(torch.round(torch.sigmoid(pred)).item())
  # Print out what's happening and convert model logits -> pred probs -> pred label
  print(f"Ouput logits: \n{pred}\n")
  print(f"Output prediction probabilities: \n{torch.sigmoid(pred)}\n")
  print(f"Ouput prediction label: \n{torch.round(torch.sigmoid(pred))}\n")
  print(f"Actual label: \n{label}")

  fig, axs = plt.subplots(1, 2)

  # Image 1 (No Grad-CAM)
  true_label = idx_class[label]
  axs[0].imshow(rgb_img)
  axs[0].set_title("Image with No Grad-CAM\n" +
                  f"Actual Label: {true_label}")

  # Image 2 (Grad-CAM)
  pred_label = idx_class[pred_label]
  axs[1].imshow(visualization)
  axs[1].set_title("Image with Grad-CAM\n" +
                  f"Pred Label: {pred_label}")
  plt.show()