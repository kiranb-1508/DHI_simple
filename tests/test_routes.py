import unittest
from flask import Flask
from flask.testing import FlaskClient

class TestFlaskApp(unittest.TestCase):
    def setUp(self):
        # Create a test client for the Flask app
        self.app = Flask(__name__)
        self.client = self.app.test_client()

    def test_flask_import(self):
        app = Flask(__name__)
        self.assertIsNotNone(app)

    def test_home_route(self):
        @self.app.route('/')
        def home():
            return 'Hello, World!'

        response = self.client.get('/')
        self.assertEqual(response.data.decode(), 'Hello, World!')
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()

