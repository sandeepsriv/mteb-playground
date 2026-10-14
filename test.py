import mteb
from sentence_transformers import SentenceTransformer

# Select model
# model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_name = "nvidia/Nemotron-3-Embed-8B-BF16"
model = mteb.get_model(
    model_name
)  # if the model is not implemented in MTEB it will be eq. to SentenceTransformer(model_name)

# Select tasks
tasks = mteb.get_tasks(tasks=["Banking77Classification.v2"])

# evaluate
results = mteb.evaluate(model, tasks=tasks)