from app import app

client = app.test_client()


def shown_label(text: str) -> str:
    html = client.post("/predict", data={"user_input": text}).get_data(as_text=True)
    return html.split('badge-value">')[1].split("<")[0]


def test_home_page_loads():
    assert client.get("/").status_code == 200


def test_positive_and_negative_reach_the_page():
    assert shown_label("I love this") == "positive"
    assert shown_label("I hate this") == "negative"


def test_empty_input_goes_back_home():
    response = client.post("/predict", data={"user_input": "   "})
    assert response.status_code == 302
