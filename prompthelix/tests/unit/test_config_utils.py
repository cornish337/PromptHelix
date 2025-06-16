import unittest
import copy # For testing non-destructive behavior
from prompthelix.utils.config_utils import update_settings

class TestUpdateSettings(unittest.TestCase):

    def test_simple_override(self):
        base = {'a': 1, 'b': 2}
        override = {'b': 3, 'c': 4}
        expected = {'a': 1, 'b': 3, 'c': 4}
        self.assertEqual(update_settings(base, override), expected)

    def test_nested_override(self):
        base = {'a': 1, 'b': {'x': 10, 'y': 20}}
        override = {'b': {'y': 30, 'z': 40}, 'c': 3}
        expected = {'a': 1, 'b': {'x': 10, 'y': 30, 'z': 40}, 'c': 3}
        self.assertEqual(update_settings(base, override), expected)

    def test_add_new_keys(self):
        base = {'a': 1}
        override = {'b': 2, 'c': {'d': 3}}
        expected = {'a': 1, 'b': 2, 'c': {'d': 3}}
        self.assertEqual(update_settings(base, override), expected)

    def test_add_new_keys_nested(self):
        base = {'a': {'b': 1}}
        override = {'a': {'c': 2, 'd': 3}, 'e': 4}
        expected = {'a': {'b': 1, 'c': 2, 'd': 3}, 'e': 4}
        # Note: The provided update_settings function does a full merge for nested dicts.
        # If 'a' in override didn't have 'b', 'b' from base would remain.
        self.assertEqual(update_settings(base, override), expected)

    def test_different_value_types(self):
        base = {'name': 'Alice', 'age': 30, 'active': True, 'scores': [1,2,3]}
        override = {'age': 31, 'active': False, 'scores': [4,5], 'city': 'New York'}
        expected = {'name': 'Alice', 'age': 31, 'active': False, 'scores': [4,5], 'city': 'New York'}
        self.assertEqual(update_settings(base, override), expected)

    def test_non_destructive_to_originals(self):
        base = {'a': 1, 'b': {'x': 10}}
        base_original = copy.deepcopy(base)
        override = {'b': {'y': 20}, 'c': 3}
        override_original = copy.deepcopy(override)

        update_settings(base, override)

        self.assertEqual(base, base_original, "Base dictionary was modified.")
        self.assertEqual(override, override_original, "Override dictionary was modified.")

    def test_empty_override_settings(self):
        base = {'a': 1, 'b': 2}
        override = {}
        expected = {'a': 1, 'b': 2}
        # The function returns a copy
        self.assertEqual(update_settings(base, override), expected)
        self.assertIsNot(update_settings(base, override), base, "Should return a copy even if override is empty.")


    def test_empty_base_settings(self):
        base = {}
        override = {'a': 1, 'b': 2}
        expected = {'a': 1, 'b': 2}
        self.assertEqual(update_settings(base, override), expected)
        self.assertIsNot(update_settings(base, override), override, "Should return a new dict, not the override dict itself.")

    def test_override_with_none_value_in_dict(self):
        base = {'a': 1, 'b': {'x': 10, 'y': 20}}
        override = {'b': {'y': None, 'z': 40}, 'c': None}
        expected = {'a': 1, 'b': {'x': 10, 'y': None, 'z': 40}, 'c': None}
        self.assertEqual(update_settings(base, override), expected)

    def test_override_settings_is_none(self):
        base = {'a': 1, 'b': 2}
        override = None
        expected = {'a': 1, 'b': 2}
        # update_settings now handles override being None
        self.assertEqual(update_settings(base, override), expected)
        self.assertIsNot(update_settings(base, override), base)


if __name__ == '__main__':
    unittest.main()
