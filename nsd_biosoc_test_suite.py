"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu
GitHub repository: https://github.com/neat-one/nsd_biosoc

Description: A unit test suit for the nsd_biosoc project.
"""
# Import packages
import unittest
class NSD_BIOSOC(unittest.TestCase):
    """
    Author: Veronica Porubsky [Github: https://github.com/vporubsky][ORCID: https://orcid.org/0000-0001-7216-3368]

    Test suite to check that the NSD_BIOSOC library functions as expected during development.
    """
    @classmethod
    def setUpClass(cls):
        """Runs before any tests have been executed."""

        pass

    @classmethod
    def tearDownClass(cls):
        "Runs after all tests have been executed."
        pass

    def setUp(self):
        """Runs before each test. """

        pass

    def tearDown(self):
        """Runs after each test."""
        pass

    def test_assertEqual(self):
        """Test raises an AssertionError because True != False. This results in a failure."""
        self.assertEqual(True, False)

    def test_assertIsNotNone(self):
        """Test raises an AssertionError because the value passed is None. This results in a failure."""
        self.assertIsNotNone(None)

    def test_assertIs(self):
        """Test raises an AssertionError because the values passed are not the same. This results in a failure."""
        self.assertIs(1,2)




if __name__ == '__main__':
    unittest.main()
