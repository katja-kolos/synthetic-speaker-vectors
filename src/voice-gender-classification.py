import sys
import torch
import os
import json
import pandas as pd


sys.path.append("../voice-gender-classifier")

from model import ECAPA_gender

# You could directly download the model from the huggingface model hub
model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
model.eval()

# If you are using gpu .... 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



# parent_folder = "/home/users1/kolosea/hiwi/thesis/synthetic-voices/Voice-Privacy-Challenge-2024/exp-conditional_diffusion-cg-v1.5/anon_pipeline_stttsAttr/anon_speech/ims_stttsAttr_pc"
# out_folder = "../../workspace/VPCgender/cg-v1.5"
parent_folder = "/home/users1/kolosea/hiwi/thesis/synthetic-voices/Voice-Privacy-Challenge-2024/exp/anon_pipeline_stttsAttr/anon_speech/ims_stttsAttr_pc" #cwgan
out_folder = "../../workspace/VPCgender/cwgan"


with torch.no_grad():
    for name in os.listdir(parent_folder):
        # folder = "/IEMOCAP_dev"
        full_path = os.path.abspath(os.path.join(parent_folder, name))
        if os.path.isdir(full_path):
            print(name)
            results = []
            for wav_file in sorted(os.listdir(full_path)):
                if wav_file.endswith("wav"):
                    wav_path = os.path.join(full_path, wav_file)
                    try:
                        gender = model.predict(wav_path, device=device)
                        results.append((wav_path, gender))
                    except Exception as e:
                        print("Cound not calculate gender: ", e)
                        results.append((wav_path, "UNKN"))

            
            df = pd.DataFrame(results)
            df.columns = ['path', 'gender']
            # df['gender'] = df['gender'].apply(lambda x: x[0])

            outfile = os.path.join(out_folder, f"{name}.csv")           
            df.to_csv(outfile)

            torch.cuda.empty_cache()