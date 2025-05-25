import unittest
import torch

# Assuming project_name is in PYTHONPATH or tests are run from root
from project_name.models.main_model import Model 

class TestMainModel(unittest.TestCase):

    def test_model_initialization_and_forward_pass(self):
        # --- Parameters for Test ---
        num_classes = 3 # Example number of classes
        batch_size = 4    # Example batch size
        img_height = 224
        img_width = 224
        num_channels = 1  # Grayscale

        # --- Model Initialization ---
        # Initialize with default hidden_layers or specify if needed for the test
        model = Model(num_classes=num_classes, hidden_layers=[256, 128]) 
        model.eval() # Set to evaluation mode for testing (disables dropout, etc.)

        # --- Dummy Input Tensor ---
        # Create a dummy input tensor: (batch_size, channels, height, width)
        dummy_input = torch.randn(batch_size, num_channels, img_height, img_width)

        # --- Forward Pass ---
        # Perform a forward pass
        with torch.no_grad(): # Disable gradient calculations for inference
            logits = model(dummy_input)

        # --- Assertions ---
        # 1. Check output type
        self.assertIsInstance(logits, torch.Tensor)

        # 2. Check output shape
        expected_shape = (batch_size, num_classes)
        self.assertEqual(logits.shape, expected_shape)

    def test_model_structure_resnet_fc_removed(self):
        # Test if the ResNet backbone's original fc layer is an Identity layer
        num_classes = 5 # Different number of classes for this test
        model = Model(num_classes=num_classes)
        
        # Check the type of the 'fc' layer of the ResNet backbone
        # self.backbone.fc should be nn.Identity after our refactoring
        self.assertIsInstance(model.backbone.fc, torch.nn.Identity,             "ResNet backbone's fc layer should be replaced by nn.Identity")

if __name__ == '__main__':
    unittest.main()
