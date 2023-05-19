import unittest
from AIsketcher.modelPipe import translate_language
import tracemalloc
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




if __name__ == '__main__':
    unittest.main()