import unittest
from model_factory import get_model_instance
from flower_classifier import Flower_Classifier
from unittest import mock
import argparse

class Get_Model_Instance_Test(unittest.TestCase):

    @mock.patch('argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(data_dir='flowers', save_dir=None, learning_rate=0.03, hidden_units=512, epochs=3, gpu=False, arch='vgg16'))
    def test(self, mock_args):
        in_args = mock_args()
        self.assertEqual('flowers', in_args.data_dir)
        self.assertIsInstance(get_model_instance(in_args), Flower_Classifier)

if __name__ == '__main__':
    unittest.main()
