
# Multilingual Translator

This project is a simple web-based translator that supports translation to Tamil, French, and Spanish. It uses Flask for the web framework, and a trained neural network model for the translation tasks.

## Features
- Translate text to Tamil, French, and Spanish.
- User-friendly web interface.
- Uses LSTM-based Seq2Seq models for translations.

## Requirements
- Flask
- Numpy
- Pickle
- scikit-learn
- TensorFlow

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/translator.git
    cd translator
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Make sure you have the trained models (`tamil.h5`, `french.h5`, `spanish.h5`) and the corresponding data files (`training_data_tamil.pkl`, `training_data_french.pkl`, `training_data_spanish.pkl`) in the root directory of the project.

## Usage
1. Run the Flask app:
    ```bash
    python app.py
    ```

2. Open your web browser and navigate to `http://127.0.0.1:5000/`.

3. Select the target language, input the text you want to translate, and hit "Translate".

## File Structure
- `app.py`: Main application file containing the Flask app and translation logic.
- `templates/`: Directory containing HTML templates.
- `training_data_tamil.pkl`, `training_data_french.pkl`, `training_data_spanish.pkl`: Pickle files containing necessary data for translation.
- `tamil.h5`, `french.h5`, `spanish.h5`: Pre-trained models for translation.

## Example
To translate "Hello" to French, select "French" from the dropdown, enter "Hello" in the text box, and click "Translate". The translated text will appear below.

## Acknowledgments
This project uses LSTM-based Seq2Seq models for translation, leveraging TensorFlow and scikit-learn libraries.


