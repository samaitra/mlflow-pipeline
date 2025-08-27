import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import time

load_dotenv()
client = OpenAI()

eval_data = pd.DataFrame(
    {
        "inputs": [
            "Artificial intelligence has transformed how businesses operate in the 21st century. Companies are leveraging AI for everything from customer service to supply chain optimization. The technology enables automation of routine tasks, freeing human workers for more creative endeavors. However, concerns about job displacement and ethical implications remain significant. Many experts argue that AI will ultimately create more jobs than it eliminates, though the transition may be challenging.",
            "Climate change continues to affect ecosystems worldwide at an alarming rate. Rising global temperatures have led to more frequent extreme weather events including hurricanes, floods, and wildfires. Polar ice caps are melting faster than predicted, contributing to sea level rise that threatens coastal communities. Scientists warn that without immediate and dramatic reductions in greenhouse gas emissions, many of these changes may become irreversible. International cooperation remains essential but politically challenging.",
            "The human genome project was completed in 2003 after 13 years of international collaborative research. It successfully mapped all of the genes of the human genome, approximately 20,000-25,000 genes in total. The project cost nearly $3 billion but has enabled countless medical advances and spawned new fields like pharmacogenomics. The knowledge gained has dramatically improved our understanding of genetic diseases and opened pathways to personalized medicine. Today, a complete human genome can be sequenced in under a day for about $1,000.",
            "Remote work adoption accelerated dramatically during the COVID-19 pandemic. Organizations that had previously resisted flexible work arrangements were forced to implement digital collaboration tools and virtual workflows. Many companies reported surprising productivity gains, though concerns about company culture and collaboration persisted. After the pandemic, a hybrid model emerged as the preferred approach for many businesses, combining in-office and remote work. This shift has profound implications for urban planning, commercial real estate, and work-life balance.",
            "Quantum computing represents a fundamental shift in computational capability. Unlike classical computers that use bits as either 0 or 1, quantum computers use quantum bits or qubits that can exist in multiple states simultaneously. This property, known as superposition, theoretically allows quantum computers to solve certain problems exponentially faster than classical computers. Major technology companies and governments are investing billions in quantum research. Fields like cryptography, material science, and drug discovery are expected to be revolutionized once quantum computers reach practical scale.",
        ],
        "targets": [
            "AI has revolutionized business operations through automation and optimization, though ethical concerns about job displacement persist alongside predictions that AI will ultimately create more employment opportunities than it eliminates.",
            "Climate change is causing accelerating environmental damage through extreme weather events and melting ice caps, with scientists warning that without immediate reduction in greenhouse gas emissions, many changes may become irreversible.",
            "The Human Genome Project, completed in 2003, mapped approximately 20,000-25,000 human genes at a cost of $3 billion, enabling medical advances, improving understanding of genetic diseases, and establishing the foundation for personalized medicine.",
            "The COVID-19 pandemic forced widespread adoption of remote work, revealing unexpected productivity benefits despite collaboration challenges, and resulting in a hybrid work model that impacts urban planning, real estate, and work-life balance.",
            "Quantum computing uses qubits existing in multiple simultaneous states to potentially solve certain problems exponentially faster than classical computers, with major investment from tech companies and governments anticipating revolutionary applications in cryptography, materials science, and pharmaceutical research.",
        ],
    }
)

import mlflow
import openai


def predict(data: pd.DataFrame) -> list[str]:
    predictions = []
    prompt = mlflow.load_prompt("prompts:/summarization-prompt/1")

    for _, row in data.iterrows():
        # Fill in variables in the prompt template
        content = prompt.format(sentences=row["inputs"], num_sentences=1)
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
        )
        predictions.append(completion.choices[0].message.content)

    return predictions

with mlflow.start_run(run_name="prompt-evaluation"):
    mlflow.log_param("model", "gpt-4o-mini")
    mlflow.log_param("temperature", 0.1)

    results = mlflow.evaluate(
        model=predict,
        data=eval_data,
        targets="targets",
        extra_metrics=[
            # Specify GPT4 as a judge model for answer similarity. Other models such as Anthropic,
            # Bedrock, Databricks, are also supported.
            mlflow.metrics.genai.answer_similarity(model="openai:/gpt-4"),
        ],
    )