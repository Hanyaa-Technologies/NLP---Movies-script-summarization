### Movie Script Summarization
This project utilizes the Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0 model to summarize movie scripts in Telugu. The model is fine-tuned to handle summarization tasks and generate concise summaries of provided text.

##### Prerequisites
Before you begin,ensure you have met the following requirements:
1. Python 3.8 or above
2. PyTorch 1.10.0 or above
3. Transformers library by Hugging Face 

##### Installation
1. Clone the repository: 
git clone <repository-url>
cd <repository-directory>

2. Install the required packages:
   pip install torch transformers

##### Usage
Setting up the necessary libraries and tools
Import the necessary libraries: 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM 

Import Model and Tokenizer
Load the model and tokenizer: 
model_name = "Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name) 

##### Script Summarization Example 

Example 1

Provide an instruction and input text for summarization: 
instruction = "కింది వచనాన్ని సంగ్రహించండి"
input_text = """అర్జున్ రెడ్డి దేశ్ ముఖ్ సంపన్న వ్యాపారవేత్త ధనుంజయ్ రెడ్డి దేశ్ ముఖ్ చిన్న కుమారుడు. అర్జున్ మంగళూరులోని సెయింట్ మేరీస్ మెడికల్ కాలేజీలో మెడికల్ స్టూడెంట్. తెలివైన విద్యార్థి అయినప్పటికీ, అతను స్కిజోటైపల్ పర్సనాలిటీ డిజార్డర్ కలిగి ఉన్నాడు, ఇది కోపం నిర్వహణ సమస్యలను కలిగిస్తుంది, ఇది కళాశాల డీన్ యొక్క ఆగ్రహాన్ని పొందుతుంది. అర్జున్ దూకుడు స్వభావం అతని జూనియర్లలో కళాశాల రౌడీగా పేరు సంపాదించింది. ఇంటర్ కాలేజ్ ఫుట్ బాల్ మ్యాచ్ సందర్భంగా ప్రత్యర్థి జట్టు సభ్యులతో తన స్నేహితుడు కమల్ తో గొడవ జరిగిన తరువాత డీన్ అర్జున్ ను క్షమాపణ చెప్పమని లేదా కళాశాలను విడిచిపెట్టమని అడుగుతాడు. అర్జున్ మొదట కళాశాలను విడిచిపెట్టాలని నిర్ణయించుకుంటాడు, కాని మొదటి సంవత్సరం విద్యార్థిని ప్రీతి శెట్టిని కలిసిన తరువాత అక్కడే ఉంటాడు."""

text = f"Instruction: {instruction} \nInput: {input_text} \nResponse:" 

Tokenize the input text: 
encodings = tokenizer(text, padding=True, return_tensors="pt") 

Generate the summary using the model: 
with torch.inference_mode():
    outputs = model.generate(encodings.input_ids, do_sample=True, max_new_tokens=500)

output = tokenizer.batch_decode(outputs.detach(), skip_special_tokens=True)
print(output)

##### Hugging face Link 
https://huggingface.co/Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0






