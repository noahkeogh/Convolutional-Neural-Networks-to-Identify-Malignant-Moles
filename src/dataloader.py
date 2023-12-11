from torchvision import datasets
from torch.utils.data import random_split
from torch.utils.data import DataLoader

def load_train_val_test(train_dir, train_transform, test_dir, test_transform, val_perc):
  """
  Loads the train, test, and validation data given 
  the paths to the train and test directories and the 
  percent of training data to split for validation. 

  Inputs: 
    train_dir: Path object to the training images directory
    train_transform: transformations to apply to the training dataset 
    test_dir: Path object to the test images directory 
    test_transform: transformations to apply to the testing dataset
    val_perc: float representing percent of the training data to save for validation 

  Outputs: 
    train_data: datasets object containing transformed train images 
    val_data: datasets object containing transformed validation images 
    test_data: datasets object containing transformed test images 
  """
  # Load the training data
  train_data = datasets.ImageFolder(root=train_dir,
                                    transform=train_transform,
                                    target_transform=None)
  # Load the testing data
  test_data = datasets.ImageFolder(root=test_dir,
                                  transform=test_transform,
                                  target_transform=None)
  # Determine size of validation and training data
  val_size = int(len(train_data) * val_perc)
  train_size = len(train_data) - val_size
  # Split the Validation Data from the training data
  train_data, val_data = random_split(train_data, [train_size, val_size])
  return train_data, val_data, test_data 
  

def load_data_into_dataloader(data, batch_size, num_workers):
    """
    Given data it will load it into the dataloader. 
    
    Inputs: 
    data: the dataset to load into the loader
    batch_size: the desired batch size 
    num_workers: number of workers to use to load data 
    
    Returns: 
    The data loaded into the dataloader. 
    """
    return DataLoader(
        dataset=data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True
        )