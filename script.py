from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model_id = "sentence-transformers/all-MiniLM-L6-v2"
save_directory = "./local_model_weights"

# Download and export to ONNX
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)

# Save both to your project folder
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)