import unittest
import sys
import os
import importlib

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestAppIntegrity(unittest.TestCase):
    def test_app_import(self):
        """Smoke test: Verify app.py can be imported without crashing."""
        try:
            import app
            self.assertTrue(hasattr(app, 'main'))
        except ImportError as e:
            self.fail(f"Failed to import app.py: {e}")
        except SyntaxError as e:
            self.fail(f"Syntax error in app.py: {e}")

if __name__ == '__main__':
    unittest.main()
