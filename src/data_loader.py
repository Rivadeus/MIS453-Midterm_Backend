import pandas as pd

def load_data():
    """
    Generates synthetic data for the midterm project.
    In a real app, this would download from Hugging Face or load a CSV.
    """
    data = {
        'text': [
            "I loved this movie, it was fantastic", 
            "Great acting and amazing plot", 
            "Worst movie I have ever seen", 
            "Terrible script and bad acting",
            "It was okay, not great but not bad",
            "A masterpiece of cinema",
            "Boring and too long",
            "I fell asleep halfway through",
            "Highly recommended",
            "Complete waste of time",
            "The direction was superb",
            "I didn't like the characters",
            "Visuals were stunning",
            "Plot holes everywhere"
        ],
        'label': ['Positive', 'Positive', 'Negative', 'Negative', 'Positive', 
                  'Positive', 'Negative', 'Negative', 'Positive', 'Negative',
                  'Positive', 'Negative', 'Positive', 'Negative']
    }
    return pd.DataFrame(data)