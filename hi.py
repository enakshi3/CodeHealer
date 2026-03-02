from safetensors.torch import load_file
import torch
from transformers import RobertaTokenizerFast, T5ForConditionalGeneration, T5Config
import json

# Paths for model configuration, generation config, and model weights
config_path = "C:/Users/mouri/Documents/VIT/Codehealer HPE/hitesh/config.json"
generation_config_path = "C:/Users/mouri/Documents/VIT/Codehealer HPE/hitesh/generation_config.json"
model_path = "C:/Users/mouri/Documents/VIT/Codehealer HPE/hitesh/model.safetensors"
test_data_path = "C:/Users/mouri/Downloads/test1.json"  

# Load tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("Salesforce/codet5-base")

# Load model configuration
model_config = T5Config.from_json_file(config_path)

# Initialize the model
model = T5ForConditionalGeneration(config=model_config)

# Load model weights with strict=False to allow missing keys
state_dict = load_file(model_path)
missing_keys = []

for key in state_dict.keys():
    if key.startswith("encoder.") or key.startswith("decoder.") or key.startswith("lm_head."):
        if key not in model.state_dict():
            missing_keys.append(key)

if missing_keys:
    print(f"Missing keys found in state_dict: {missing_keys}")

model.load_state_dict(state_dict, strict=False)

# Move model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

# Load test data
with open(test_data_path) as f:
    test_data = json.load(f)

# Initialize counters
total_cases = 0
correct_fixes = 0

# Function to generate predictions
def predict_fix(code):
    inputs = tokenizer(code, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Evaluate model accuracy
for case in test_data['data']:
    original_src = case['original_src']
    expected_fix = case['changed_src']
    
    # Get model's predicted fix
    predicted_fix = predict_fix(original_src)
    
    # Increment total cases
    total_cases += 1
    
    # Check if model's prediction matches the expected fix
    if predicted_fix.strip() == expected_fix.strip():
        correct_fixes += 1

# Calculate accuracy
accuracy = (correct_fixes / total_cases)
print(f"Model Accuracy: {accuracy}")
