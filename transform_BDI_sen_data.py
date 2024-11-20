import os
import argparse
from tqdm import tqdm, trange
import pandas as pd
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    args = parser.parse_args()

    Symptoms = ['Sadness', 'Pessimism', 'Sense_of_failure', 'Loss_of_Pleasure', 'Guilty_feelings', 'Sense_of_punishment', 'Self-dislike', 'Self-incrimination', 'Suicidal_ideas', 'Crying', 'Agitation', 'Social_withdrawal', 'Indecision', 'Feelings_of_worthlessness', 'Loss_of_energy', 'Change_of_sleep', 'Irritability', 'Changes_in_appetite', 'Concentration_difficulty', 'Tiredness_or_fatigue', 'Loss_of_interest_in_sex']

    for filename in os.listdir(args.input_path):
        if "multilabels" in filename:
            output_list = []
            df = pd.read_csv(os.path.join(args.input_path, filename))
            output = {}
            for idx in trange(len(df), desc="Processing"+filename):
                output["text"] = df['Sentence'][idx]
                output["symptoms"] = []
                output["label_id"] = ''
                for symptom in Symptoms:
                    output["label_id"] += str(df[symptom][idx])
                    if df[symptom][idx] == 1:
                        output["symptoms"].append(symptom)
                output['label_id'] = int(output['label_id'])
                output_list.append(output.copy())

            with open('Dataset/'+filename.split("-")[0] + "_BDISen.json", "w+") as f:
                for line in output_list:
                    f.write(json.dumps(line) + "\n")