import unittest
from AIsketcher.modelPipe import translate_language, resize_image
import tracemalloc
from PIL import Image
import numpy as np
import os

tracemalloc.start()
class TestAIskectcher(unittest.TestCase):

    def test_translate_test_kr(self):
        test_input = '꿈, 상상속의 나라, 만화, 특이한 캐릭터, 이상한, 멋진, 귀여운, 환상적인'
        print('test_input : ', test_input)
        trans_info = {
            'region_name': 'us-east-1',
            'aws_access_key_id': '',
            'aws_secret_access_key': '',
            'SourceLanguageCode': 'ko',
            'TargetLanguageCode': 'en',
            'iam_access': False
        }
        result = translate_language(test_input,trans_info)
        print('result : ', result)

    def test_resize_image(self):
        current_directory = os.getcwd()
        print(current_directory)
        file_name = '../pic/test2.jpeg'
        image = resize_image(file_name, 800)
        image.save("result_test.jpeg")



if __name__ == '__main__':
    unittest.main()