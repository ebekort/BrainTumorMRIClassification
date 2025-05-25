import unittest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
from PIL import Image # Required for type hinting if Image.open is mocked carefully
import os # Import os for os.path.join

# Assuming project_name is in PYTHONPATH or tests are run from root
from project_name.utils.dataset import MRI_Dataset
from torchvision import transforms

class TestMRIDataset(unittest.TestCase):

    @patch('project_name.utils.dataset.os.listdir')
    @patch('project_name.utils.dataset.Image.open')
    def test_dataset_initialization_and_loading(self, mock_image_open, mock_os_listdir):
        # --- Mocking Setup ---
        # Mock os.listdir to return a list of class directories
        mock_os_listdir.side_effect = lambda path: {
            './data/dummy_dataset': ['classA', 'classB'], # For base directory
            os.path.join('./data/dummy_dataset', 'classA'): ['img1.png', 'img2.png'], # For classA
            os.path.join('./data/dummy_dataset', 'classB'): ['img3.png']  # For classB
        }.get(path, []) # Default to empty list if path not specified

        # Mock Image.open().convert() to return a dummy PIL Image-like object
        mock_pil_image = MagicMock(spec=Image.Image)
        mock_pil_image.convert.return_value = mock_pil_image 
        mock_pil_image.mode = 'L' 
        mock_pil_image.size = (10, 10) # dummy size, ToTensor will use this

        mock_image_open.return_value = mock_pil_image

        # --- Test Execution ---
        data_dir = './data/dummy_dataset'
        classes = ['classA', 'classB']
        
        test_transform = transforms.ToTensor()

        dataset = MRI_Dataset(data_dir=data_dir, classes=classes, transform=test_transform)

        # --- Assertions ---
        self.assertEqual(len(dataset), 3) # 2 from classA, 1 from classB
        
        # Check that os.listdir was called for data_dir and its subdirectories
        mock_os_listdir.assert_any_call(data_dir)
        mock_os_listdir.assert_any_call(os.path.join(data_dir, 'classA'))
        mock_os_listdir.assert_any_call(os.path.join(data_dir, 'classB'))
        
        # Test __getitem__ for the first item
        # The order of items depends on listdir mock and internal sorting if any.
        # Let's assume classA/img1.png is the first item based on mock.
        img, label = dataset[0] 
        
        self.assertIsInstance(img, torch.Tensor)
        self.assertEqual(img.ndim, 3) 
        self.assertEqual(img.shape[0], 1) # Channel = 1 for grayscale
        self.assertEqual(img.shape[1], 10) # Height from mock_pil_image.size
        self.assertEqual(img.shape[2], 10) # Width from mock_pil_image.size

        self.assertIsInstance(label, np.ndarray) 
        self.assertEqual(len(label), len(classes))
        # For the first image from 'classA' (index 0 for classes list)
        np.testing.assert_array_equal(label, np.array([1.0, 0.0]))
        
        mock_image_open.assert_any_call(os.path.join(data_dir, 'classA', 'img1.png'))


    @patch('project_name.utils.dataset.os.listdir')
    @patch('project_name.utils.dataset.Image.open')
    def test_getitem_returns_correct_types(self, mock_image_open, mock_os_listdir):
        # Setup Mocks
        mock_os_listdir.side_effect = lambda path: {
            './d': ['clA'], 
            os.path.join('./d', 'clA'): ['i1.png']
        }.get(path, [])
        
        mock_pil_image = MagicMock(spec=Image.Image)
        mock_pil_image.convert.return_value = mock_pil_image
        mock_pil_image.mode = 'L'
        mock_pil_image.size = (10,10)
        mock_image_open.return_value = mock_pil_image

        dataset = MRI_Dataset(data_dir='./d', classes=['clA'], transform=transforms.ToTensor())
        
        self.assertTrue(len(dataset) > 0, "Dataset should not be empty with mocks")
        img, label = dataset[0]

        self.assertIsInstance(img, torch.Tensor)
        self.assertIsInstance(label, np.ndarray) 
        self.assertEqual(label.dtype, np.float64) 
        self.assertEqual(img.dtype, torch.float32)


if __name__ == '__main__':
    unittest.main()
