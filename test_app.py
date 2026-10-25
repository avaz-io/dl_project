"""Headless check of app.py with Streamlit's test runner.

    python test_app.py

Loads the app, lists the models, and runs one small (32x32) and one large example through
the default model, failing on any exception.
"""
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app.py", default_timeout=180).run()
assert not at.exception, at.exception
models = at.selectbox[0].options
print("models listed:", len(models))
print("default model:", models[0])
for src in ["CIFAKE fake (32×32)", "Midjourney"]:
    at.selectbox[1].select(src).run()
    at.button[0].click().run()
    assert not at.exception, at.exception
    print(f"example {src!r}: {[h.value for h in at.subheader][:1]} | warning boxes: {len(at.error) + len(at.warning)}")
print("OK")
