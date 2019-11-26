import unittest

from ri_topics.util import force_trailing_slash, subpath_join


class TestForceTrailingSlash(unittest.TestCase):
    def test_is_appended_if_not_present(self):
        self.assertEqual(
            'https://www.example.com/base/',
            force_trailing_slash('https://www.example.com/base'),
        )

    def test_is_not_duplicated_if_present(self):
        self.assertEqual(
            'https://www.example.com/base/',
            force_trailing_slash('https://www.example.com/base/'),
        )


class TestSubpathJoin(unittest.TestCase):
    def test_is_appended(self):
        expected = 'https://www.example.com/base/sub1/sub2'

        self.assertEqual(expected, subpath_join('https://www.example.com/base', 'sub1/sub2'))
        self.assertEqual(expected, subpath_join('https://www.example.com/base/', 'sub1/sub2'))
        self.assertEqual(expected, subpath_join('https://www.example.com/base', '/sub1/sub2'))
        self.assertEqual(expected, subpath_join('https://www.example.com/base/', '/sub1/sub2'))


if __name__ == '__main__':
    unittest.main()
