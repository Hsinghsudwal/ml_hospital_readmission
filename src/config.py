import yaml

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

config = load_config()

# test = (config["data"]["raw_data_path"])
# registry = S3Registry(
#     bucket_name=config["localstack"]["s3_bucket"],
#     endpoint_url=config["localstack"]["endpoint_url"]
# )
