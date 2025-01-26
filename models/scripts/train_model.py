import layoutparser as lp
from detectron2.engine import DefaultTrainer


# Define the configuration
config = lp.Detectron2Config(
    label_map={1: "FieldName1", 2: "FieldName2"},  # Map labels to fields
    config_path="path/to/config.yaml",
    model_path="path/to/pretrained/model.pkl",
    extra_options={"output_dir": "models/trained"}
)

trainer = DefaultTrainer(config)
trainer.resume_or_load(resume=False)
trainer.train()
