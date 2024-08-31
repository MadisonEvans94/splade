from openai import OpenAI
import os
import random
import click

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Define the directory where documents will be saved
save_directory = './SOURCE_DOCUMENTS'

# Create the directory if it does not exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Define the 10 important areas and their subtopics for the NAVLE
navle_topics = {
    "Small Animal Medicine": ["Parasites in dogs", "Feline leukemia virus", "Diabetes mellitus in cats", "Canine osteoarthritis"],
    # "Large Animal Medicine": ["Lameness in horses", "Bovine respiratory disease", "Mastitis in dairy cows", "Reproductive disorders in cattle"],
    # "Preventive Medicine and Public Health": ["Rabies vaccination protocols", "Food safety in public health", "Epidemiology of zoonotic diseases"],
    # "Pharmacology and Therapeutics": ["Antibiotic stewardship in veterinary practice", "Anesthetics in small animals", "Anti-inflammatory drugs"],
    # "Anatomy and Physiology": ["Neuroanatomy of dogs", "Cardiovascular physiology of cats", "Digestive anatomy of ruminants"],
    "Diagnostics and Imaging": ["Interpretation of radiographs", "Blood work analysis in horses", "Principles of ultrasound in veterinary practice"],
    "Pathology and Clinical Pathology": ["Common neoplasms in dogs", "Hematology panels", "Inflammatory processes"],
    # "Emergency and Critical Care": ["Management of shock", "Toxicology in small animals", "Emergency trauma surgery"],
    # "Reproduction and Obstetrics": ["Dystocia management", "Artificial insemination in cattle", "Pregnancy diagnosis in horses"],
    # "Ethics, Laws, and Professional Practice": ["Informed consent in veterinary medicine", "Confidentiality regulations", "Animal welfare laws"],
}


def generate_document(topic, subtopic):
    """Generate a factual document using OpenAI API."""
    prompt = f"Write a 500-word factual document on the sub-topic '{subtopic}' under the topic '{topic}'. The content should be relevant for the North American Veterinary Licensing Examination (NAVLE). All content should be FACTUAL"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Specify the chat model
        messages=[{"role": "system", "content": prompt}],
        max_tokens=2048,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()


def save_document(text, topic, subtopic, index):
    """Save the generated document to a text file."""
    filename = f"{topic.replace(' ', '_')}_{subtopic.replace(' ', '_')}_{index}.txt"
    filepath = os.path.join(save_directory, filename)

    with open(filepath, 'w') as f:
        f.write(text)

    print(f"Document '{filename}' saved.")


@click.command()
@click.option('--k', default=10, help='Number of documents to generate.')
def main(k):
    """Main function to generate K documents."""
    for i in range(k):
        # Select a random topic and subtopic
        topic = random.choice(list(navle_topics.keys()))
        subtopic = random.choice(navle_topics[topic])

        # Generate document using OpenAI API
        document_text = generate_document(topic, subtopic)

        # Save document to file
        save_document(document_text, topic, subtopic, i + 1)


if __name__ == "__main__":
    main()
