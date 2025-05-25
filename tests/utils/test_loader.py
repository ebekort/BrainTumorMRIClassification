import unittest
from unittest.mock import patch, MagicMock, call # Add 'call' for checking transform assignment
import torch
from torch.utils.data import DataLoader, Subset # Subset might be needed for assertions
from torchvision import transforms # For transform instances
import os # For os.path.join

# Assuming project_name is in PYTHONPATH or tests are run from root
from project_name.utils.loader import get_dataloaders
# MRI_Dataset is imported by loader, so we patch it there.

class TestGetDataLoaders(unittest.TestCase):

    @patch('project_name.utils.loader.MRI_Dataset') # Patch MRI_Dataset in the loader's namespace
    def test_get_dataloaders_functionality(self, MockMRIDataset):
        # --- Mocking Setup ---
        # Configure the mock MRI_Dataset class
        mock_dataset_instance = MagicMock()
        
        # Simulate 100 samples: 60 class 0, 40 class 1 for stratification testing
        mock_dataset_instance.dataset = [('path/to/img' + str(i), 0) for i in range(60)] + \
                                        [('path/to/img' + str(i), 1) for i in range(60, 100)]
        
        def get_len():
            return len(mock_dataset_instance.dataset)
        mock_dataset_instance.__len__ = get_len

        # Make the constructor of MockMRIDataset return our instance
        MockMRIDataset.return_value = mock_dataset_instance

        # --- Test Execution ---
        data_dir = './fake_data_dir' # This won't be used by the mocked MRI_Dataset
        classes = ['class0', 'class1'] # Needs to match labels in mock_dataset_instance.dataset
        batch_size = 10
        # test_size=0.15, val_size=0.1765 (approx 15% of remaining 85%)
        # For 100 samples: test=15, remaining=85. val=0.1765*85 approx 15. train=70.
        
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir=data_dir, 
            classes=classes, 
            batch_size=batch_size,
            test_size=0.15, # 15 samples for test
            val_size=0.1765, # val_size from (1-test_size) pool -> 0.15 * 0.85 = approx 15 samples
            random_state=42,
            num_workers=0 # Easier for testing
        )

        # --- Assertions ---
        # 1. Check if MRI_Dataset was instantiated multiple times (for train, val_master, test_master, and total_samples_dataset)
        #    The actual number of calls might be 4 due to total_samples_dataset, train_dataset_full, val_dataset_master, test_dataset_master
        self.assertTrue(MockMRIDataset.call_count >= 3) 

        # 2. Check transform assignments (this is a bit more involved)
        #    The loader creates separate train_transform and val_test_transform.
        #    It then calls MRI_Dataset with these.
        #    Example: Check that the first call (total_samples_dataset) has transform=None
        #    And subsequent calls have specific transform objects.
        #    For simplicity here, we'll focus on dataloader properties.
        #    More advanced: check args_list of MockMRIDataset.
        
        # Check calls to MRI_Dataset constructor with correct transforms
        # First call is for total_samples_dataset with transform=None
        # Second call is for train_dataset_full with a training transform
        # Third call is for val_dataset_master with a validation/test transform
        # Fourth call is for test_dataset_master with a validation/test transform
        
        # We can inspect the 'transform' argument in the calls to the mock
        # This requires that the transforms are not identical objects if we want to distinguish them.
        # The loader creates train_transform and val_test_transform.
        
        # Get the actual transform objects created within get_dataloaders to compare
        # This is hard without refactoring get_dataloaders to return them or make them global.
        # Alternative: Check properties of the datasets within the loaders, if possible.
        # For now, let's verify the DataLoaders themselves.

        # 3. Check if three DataLoader objects are returned
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        self.assertIsInstance(test_loader, DataLoader)

        # 4. Check approximate dataset sizes in DataLoaders (after Subset splitting)
        #    Need to access the dataset within the DataLoader and then its length
        #    len(loader.dataset) gives the length of the Subset
        self.assertEqual(len(train_loader.dataset), 70) # Expected 70 training samples
        self.assertEqual(len(val_loader.dataset), 15)   # Expected 15 validation samples
        self.assertEqual(len(test_loader.dataset), 15)  # Expected 15 test samples
        
        # 5. Check batch_size
        self.assertEqual(train_loader.batch_size, batch_size)
        self.assertEqual(val_loader.batch_size, batch_size)
        self.assertEqual(test_loader.batch_size, batch_size)

        # 6. Stratification check (simplified):
        #    Ensure all indices are unique and cover the range 0-99
        train_indices = train_loader.dataset.indices
        val_indices = val_loader.dataset.indices
        test_indices = test_loader.dataset.indices
        
        all_indices_from_subsets = sorted(train_indices + val_indices + test_indices)
        self.assertEqual(all_indices_from_subsets, list(range(100)))


if __name__ == '__main__':
    unittest.main()
