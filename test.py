from torch.utils.data import DataLoader, TensorDataset
from transformers import XLMRobertaTokenizer, XLMRobertaForCausalLM, AdamW

# Step 2: Load your dataset
# Assuming your dataset is loaded into three lists: questions, answers, and user_queries.
user_queries = total_data['query']
questions = total_data['question']
answers = total_data['answer']

# Step 3: Preprocess the data
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

input_ids = []
attention_masks = []
labels = []

input_texts = [f"<Q>{question}<A>{answer}" for question, answer in zip(questions, answers)]
label_texts = user_queries

input_encodings = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
label_encodings = tokenizer(label_texts, return_tensors="pt", padding=True, truncation=True)

input_ids = input_encodings["input_ids"]
attention_masks = input_encodings["attention_mask"]
labels = label_encodings["input_ids"]

max_sequence_length = 12  # You can set this to your desired maximum length

# Pad or truncate sequences to the maximum length
input_ids = input_encodings['input_ids'][:, :max_sequence_length]
attention_masks = input_encodings['attention_mask'][:, :max_sequence_length]
labels = label_encodings['input_ids'][:, :max_sequence_length]

dataset = TensorDataset(input_ids, attention_masks, labels)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Step 4: Fine-tune the model
model = XLMRobertaForCausalLM.from_pretrained("xlm-roberta-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch in data_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        # Ensure labels have the same batch size as input_ids and attention_mask
        labels = labels[:, :input_ids.shape[1]]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
# Step 5: Model is now fine-tuned

# Step 6: Generate user queries using the fine-tuned model
def generate_user_query(question, answer):
    input_text = f"<Q>{question}<A>{answer}"
    input_encoding = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
    generated = model.generate(input_ids=input_encoding["input_ids"], attention_mask=input_encoding["attention_mask"])
    generated_query = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_query

# Example usage:
sample_question = "How do I install software?"
sample_answer = "You can install software by following these steps..."
generated_query = generate_user_query(sample_question, sample_answer)
print(f"Generated User Query: {generated_query}")