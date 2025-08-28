from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get the latest registered version (autologging creates version 1)
model_version = client.get_registered_model("iris_classifier").latest_versions[0]

# Set production alias (replaces deprecated stages)
client.set_registered_model_alias(
    name="iris_classifier", alias="production", version=model_version.version
)

print(f"Model version {model_version.version} tagged as 'production'")

# Model URI for serving (using alias)
model_uri = "models:/iris_classifier@production"
print(f"Production model URI: {model_uri}")