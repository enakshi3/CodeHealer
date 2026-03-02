from flask import Flask, request, jsonify, render_template
from safetensors.torch import load_file
import torch
from transformers import RobertaTokenizerFast, T5ForConditionalGeneration, T5Config

app = Flask(__name__)

config_path = "C:/Users/mouri/Documents/VIT/Codehealer HPE/hitesh/config.json"
generation_config_path = "C:/Users/mouri/Documents/VIT/Codehealer HPE/hitesh/generation_config.json"

model_config = T5Config.from_json_file(config_path)

model = T5ForConditionalGeneration(config=model_config)

model_path = "C:/Users/mouri/Documents/VIT/Codehealer HPE/hitesh/model.safetensors"
state_dict = load_file(model_path)

missing_keys = []
for key in state_dict.keys():
    if key.startswith("encoder.") or key.startswith("decoder.") or key.startswith("lm_head."):
        if key not in model.state_dict():
            missing_keys.append(key)

if missing_keys:
    print(f"Missing keys found in state_dict: {missing_keys}")

model.load_state_dict(state_dict, strict=False)

model.eval()

tokenizer = RobertaTokenizerFast.from_pretrained("Salesforce/codet5-base")

@app.route('/')
def index():
    return render_template('index.html')

def generate_code_suggestions(tokenizer_instance, model_instance, error_message, original_code_snippet, beam_search_size=4):
    tokenized_inputs = tokenizer_instance(
        text=error_message, 
        text_pair=original_code_snippet, 
        max_length=512, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    ).to(model_instance.device)
    
    with torch.no_grad():
        generated_code_ids = model_instance.generate(
            input_ids=tokenized_inputs['input_ids'],
            attention_mask=tokenized_inputs['attention_mask'],
            num_beams=beam_search_size, 
            no_repeat_ngram_size=2, 
            num_return_sequences=beam_search_size, 
            max_length=512
        ).cpu().detach().numpy()

    first_suggestion = tokenizer_instance.decode(generated_code_ids[0], skip_special_tokens=True)
    
    return [first_suggestion]

@app.route('/correct_code', methods=['POST'])
def correct_code_route():
    data = request.get_json()
    code = data['code']
    error = data['error']
    suggestions = generate_code_suggestions(tokenizer, model, error, code)
    return jsonify({'suggestions': suggestions})

if __name__ == '__main__':
    app.run(debug=True)