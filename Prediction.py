svm_preds = final_svm_model.predict(test_X)
nb_preds = final_nb_model.predict(test_X)
rf_preds = final_rf_model.predict(test_X)

svm_preds = svm_preds.tolist()
nb_preds = nb_preds.tolist()
rf_preds = rf_preds.tolist()

print("svm_preds:", svm_preds)
print("nb_preds:", nb_preds)
print("rf_preds:", rf_preds)

print(len(svm_preds), len(nb_preds), len(rf_preds))

print(svm_preds)
print(nb_preds)
print(rf_preds)

final_preds = [mode([i, j, k])[0] for i, j, k in zip(svm_preds, nb_preds, rf_preds)]

print(f"Accuracy on Test dataset by the combined model: {accuracy_score(test_Y, final_preds)*100}")

cf_matrix = confusion_matrix(test_Y, final_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Combined Model on Test Dataset")
plt.show()

symptoms = X.columns.values
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}

def predictDisease(symptoms):
    symptoms = symptoms.split(",")
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"].get(symptom.capitalize())
        if index is not None:
            input_data[index] = 1

    input_data = np.array(input_data).reshape(1, -1)
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]

    final_prediction = max(set([rf_prediction, nb_prediction, svm_prediction]), key=[rf_prediction, nb_prediction, svm_prediction].count)
    return final_prediction
