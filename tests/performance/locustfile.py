from locust import HttpUser, between, task

class LoadTestUser(HttpUser):
    wait_time = between(1, 2)  # Simulate a delay between requests

    @task
    def test_root(self):
        """Test the root endpoint."""
        self.client.get("/")

    @task
    def test_predict(self):
        """Test the prediction endpoint with a valid image."""
        with open("tests/data/rotten_sample.jpg", "rb") as f:
            self.client.post("/predict/", files={"file": ("rotten_sample.jpg", f, "image/jpeg")})
