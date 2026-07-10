import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class DashboardSmokeTests(unittest.TestCase):
    def test_dashboard_renders_without_streamlit_exceptions(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20)
        app.run()

        self.assertEqual(list(app.exception), [])
        self.assertEqual(
            app.title[0].value,
            "2026 FIFA World Cup Tournament Prediction Dashboard",
        )
        self.assertEqual(len(app.tabs), 5)


if __name__ == "__main__":
    unittest.main()
