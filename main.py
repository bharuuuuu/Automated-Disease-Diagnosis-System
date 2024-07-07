def main():
    sample_input = "muscle_weakness, stiff_neck, swelling_joints, movement_stiffness, painful_walking"
    predictions = predictDisease(sample_input)
    print("Predictions:")
    print("Final Prediction:", predictions)

if __name__ == "__main__":
    main()

pass

@app.route('/', methods=['GET', 'POST'])
def home():
    global chat_history  # Allow access to the global chat_history list
    
    if request.method == 'POST':
        symptoms = request.form['symptoms']
        final_prediction = predictDisease(symptoms)
        message = f"You: {symptoms}"  # Add user input to the chat history
        chat_history.append({'content': message, 'type': 'sent'})
        message = f"🤖: Based on your symptoms, it seems like you might have {final_prediction} disease. Please consult a doctor for confirmation."
        chat_history.append({'content': message, 'type': 'received'})
        return render_template('index.html', chat_history=chat_history)
    else:
        chat_history.append({'content': '🤖: Hello! Describe your symptoms, and let\'s predict potential diseases together. Your health matters!!', 'type': 'received'})
        return render_template('index.html', chat_history=chat_history)

if __name__ == '__main__':
    app.run(debug=True) 
